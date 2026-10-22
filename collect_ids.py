"""arXiv 목록에서 ID를 모으고, 초록 페이지에서 논문 정보를 추출한다."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

DEFAULT_LIST_URL = "https://arxiv.org/list/cs.AI/recent"
ABS_URL = "https://arxiv.org/abs/2301.00001"
USER_AGENT = "Educational Crawler(contact:jihunyang@naver.com)"
PAPER_CSV = Path(__file__).resolve().parent / "paper.csv"
CRAWL_LOG = Path(__file__).resolve().parent / "crawl.log"
CRAWL_DELAY_SECONDS = 15


class PaperParseError(Exception):
    """초록 페이지에서 논문 필드를 읽지 못했을 때 발생한다."""

_SUBMITTED_RE = re.compile(r"Submitted on ([^(\]]+)")

# dt 텍스트의 "arXiv:2301.00001" 또는 구형식 "arXiv:cs/0123456"
_ARXIV_ID_RE = re.compile(
    r"arXiv:(\d{4}\.\d{4,5}|[a-zA-Z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
)


def collect_arxiv_ids(url: str = DEFAULT_LIST_URL, timeout: float = 30) -> list[str]:
    """목록 페이지를 받아 등장 순서대로 arXiv ID 리스트를 반환한다.

    예: ["2610.30072", "2610.30045"]
    """
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_arxiv_ids(response.text)


def parse_arxiv_ids(html: str) -> list[str]:
    """HTML의 dt 태그 텍스트에서 arXiv ID를 추출한다."""
    soup = BeautifulSoup(html, "lxml")
    ids: list[str] = []
    seen: set[str] = set()

    for dt in soup.find_all("dt"):
        text = dt.get_text(" ", strip=True)
        for match in _ARXIV_ID_RE.finditer(text):
            arxiv_id = match.group(1)
            if arxiv_id not in seen:
                seen.add(arxiv_id)
                ids.append(arxiv_id)

    return ids


def fetch_paper_title(url: str = ABS_URL, timeout: float = 30) -> str:
    """초록 페이지를 받아 h1.title에서 논문 제목을 추출한다."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_paper_title(response.text)


def parse_paper_title(html: str) -> str:
    """HTML의 h1.title 태그에서 제목 문자열을 반환한다."""
    soup = BeautifulSoup(html, "lxml")
    return _text_after_label(soup.select_one("h1.title"), "Title")


def extract_paper(url: str = ABS_URL, timeout: float = 30) -> dict[str, str | list[str]]:
    """초록 페이지에서 논문 정보를 추출해 딕셔너리로 반환한다."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_paper(response.text, url)


def parse_paper(html: str, url: str) -> dict[str, str | list[str]]:
    """초록 페이지 HTML에서 제목, 저자, 초록, 제출일, 분야를 추출한다."""
    soup = BeautifulSoup(html, "lxml")
    title_element = soup.select_one("h1.title")
    if title_element is None:
        raise PaperParseError("h1.title 태그를 찾지 못했습니다")

    arxiv_id = _arxiv_id_from_url(url)
    dateline = _text(soup.select_one("div.dateline"))
    submitted_match = _SUBMITTED_RE.search(dateline)
    authors = [anchor.get_text(strip=True) for anchor in soup.select("div.authors a")]
    categories = [
        subject.get_text(" ", strip=True) for subject in soup.select("span.primary-subject")
    ]

    return {
        "arxiv_id": arxiv_id,
        "title": _text_after_label(title_element, "Title"),
        "authors": authors,
        "abstract": _text_after_label(soup.select_one("blockquote.abstract"), "Abstract"),
        "submitted_date": submitted_match.group(1).strip() if submitted_match else "",
        "categories": categories[0] if len(categories) == 1 else categories,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def load_saved_ids(output_path: str | Path = PAPER_CSV) -> set[str]:
    """paper.csv가 있으면 이미 저장된 arXiv ID를 반환한다."""
    path = Path(output_path)
    if not path.is_file():
        return set()

    try:
        saved = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    if "arxiv_id" not in saved.columns:
        return set()

    return {
        arxiv_id.strip()
        for arxiv_id in saved["arxiv_id"].dropna()
        if arxiv_id.strip()
    }


def crawl_papers(
    arxiv_ids: list[str],
    delay: float = CRAWL_DELAY_SECONDS,
    timeout: float = 30,
    log_path: str | Path = CRAWL_LOG,
    on_status: Callable[..., None] | None = None,
) -> list[dict[str, str | list[str]]]:
    """여러 논문 초록 페이지를 15초 간격을 두고 크롤링한다.

    HTTP 오류, 네트워크 오류, 파싱 오류가 나도 다음 논문으로 넘어간다.
    성공한 논문만 반환하고, 시도마다 crawl.log에 시각과 결과를 남긴다.
    """
    papers: list[dict[str, str | list[str]]] = []
    total = len(arxiv_ids)

    for index, arxiv_id in enumerate(arxiv_ids, start=1):
        remaining = (total - index + 1) * delay
        _notify(
            on_status,
            f"[{index}/{total}] {arxiv_id} 크롤링 시작, 남은 시간 약 {_format_remaining(remaining)}",
            index - 1,
            total,
        )
        if delay > 0:
            time.sleep(delay)

        url = f"https://arxiv.org/abs/{arxiv_id}"
        try:
            paper = extract_paper(url, timeout=timeout)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else ""
            _record_crawl(log_path, arxiv_id, "http_error", f"{status_code} {exc}".strip())
            _notify(on_status, f"[{index}/{total}] {arxiv_id} HTTP 오류: {status_code}", index, total)
        except requests.RequestException as exc:
            _record_crawl(log_path, arxiv_id, "network_error", str(exc))
            _notify(on_status, f"[{index}/{total}] {arxiv_id} 네트워크 오류: {exc}", index, total)
        except Exception as exc:
            _record_crawl(log_path, arxiv_id, "parse_error", str(exc))
            _notify(on_status, f"[{index}/{total}] {arxiv_id} 파싱 오류: {exc}", index, total)
        else:
            _record_crawl(log_path, arxiv_id, "success", str(paper["title"]))
            papers.append(paper)
            _notify(
                on_status,
                f"[{index}/{total}] {arxiv_id} 완료: {paper['title']}",
                index,
                total,
            )

    return papers


def extract_recent_papers(
    limit: int = 20,
    list_url: str = DEFAULT_LIST_URL,
    delay: float = CRAWL_DELAY_SECONDS,
    output_path: str | Path = PAPER_CSV,
    log_path: str | Path = CRAWL_LOG,
    on_status: Callable[..., None] | None = None,
) -> list[dict[str, str | list[str]]]:
    """최신 목록에서 아직 저장되지 않은 논문만 안전하게 크롤링한다."""
    _notify(on_status, f"목록 페이지 요청 전 {int(delay)}초 대기")
    if delay > 0:
        time.sleep(delay)

    arxiv_ids = collect_arxiv_ids(list_url)[:limit]
    saved_ids = load_saved_ids(output_path)
    new_ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in saved_ids]
    skipped = len(arxiv_ids) - len(new_ids)
    _notify(on_status, f"{len(arxiv_ids)}개 중 {skipped}개는 이미 있어서 건너뜀")
    return crawl_papers(new_ids, delay=delay, log_path=log_path, on_status=on_status)


def save_paper_csv(
    papers: dict[str, str | list[str]] | list[dict[str, str | list[str]]],
    output_path: str | Path = PAPER_CSV,
) -> Path:
    """새 논문을 기존 paper.csv 뒤에 붙여 저장한다."""
    rows = papers if isinstance(papers, list) else [papers]
    output = Path(output_path)
    if not rows:
        return output

    new_rows = pd.DataFrame([_flatten_paper(paper) for paper in rows])
    if output.is_file():
        try:
            existing = pd.read_csv(output, dtype=str)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        if not existing.empty:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["arxiv_id"], keep="first")
        else:
            combined = new_rows
    else:
        combined = new_rows

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _notify(
    on_status: Callable[..., None] | None,
    message: str,
    step: int | None = None,
    total: int | None = None,
) -> None:
    if on_status is None:
        print(message, flush=True)
        return
    on_status(message, step, total)


def _record_crawl(log_path: str | Path, arxiv_id: str, status: str, detail: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail = detail.replace("\n", " ").strip()
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{crawled_at}\t{status}\t{arxiv_id}\t{detail}\n")


def _format_remaining(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}분 {secs}초"
    return f"{secs}초"


def _flatten_paper(paper: dict[str, str | list[str]]) -> dict[str, str]:
    row = {key: value for key, value in paper.items()}
    if isinstance(row.get("authors"), list):
        row["authors"] = ", ".join(row["authors"])
    if isinstance(row.get("categories"), list):
        row["categories"] = ", ".join(row["categories"])
    return row


def _arxiv_id_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    arxiv_id = path.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", arxiv_id)


def _text(element) -> str:
    if element is None:
        return ""
    return element.get_text(" ", strip=True)


def _text_after_label(element, label: str) -> str:
    text = _text(element)
    prefix = f"{label}:"
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


if __name__ == "__main__":
    papers = extract_recent_papers(limit=20)
    saved = save_paper_csv(papers)
    print(f"새로 추가 {len(papers)}건, 저장 위치 {saved}")
