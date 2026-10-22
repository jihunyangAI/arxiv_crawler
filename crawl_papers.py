"""수집한 arXiv ID로 논문 초록 페이지를 크롤링한다."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from collect_ids import DEFAULT_LIST_URL, USER_AGENT, collect_arxiv_ids

# https://arxiv.org/robots.txt 의 Crawl-delay
CRAWL_DELAY_SECONDS = 15
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "cs_AI_recent.csv"

COLUMNS = [
    "arxiv_id",
    "title",
    "authors",
    "abstract",
    "subjects",
    "submitted",
    "comments",
    "abs_url",
    "pdf_url",
]

_SUBMITTED_RE = re.compile(r"Submitted on ([^(\]]+)")


def crawl_papers(
    arxiv_ids: list[str],
    delay: float = CRAWL_DELAY_SECONDS,
    timeout: float = 30,
) -> pd.DataFrame:
    """ID 목록의 초록 페이지를 순서대로 크롤링해 DataFrame으로 반환한다."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    rows: list[dict[str, str]] = []

    for index, arxiv_id in enumerate(arxiv_ids):
        if index > 0 and delay > 0:
            time.sleep(delay)
        row = fetch_paper(session, arxiv_id, timeout=timeout, delay=delay)
        rows.append(row)
        label = row["title"] or row["error"]
        print(f"[{index + 1}/{len(arxiv_ids)}] {arxiv_id} {label}")

    return pd.DataFrame(rows, columns=[*COLUMNS, "error"])


def fetch_paper(
    session: requests.Session,
    arxiv_id: str,
    timeout: float = 30,
    delay: float = CRAWL_DELAY_SECONDS,
) -> dict[str, str]:
    """초록 페이지 하나를 받아 논문 필드를 반환한다. 실패하면 한 번 재시도한다."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    last_error = ""

    for attempt in range(2):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return parse_paper(response.text, arxiv_id)
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt == 0 and delay > 0:
                time.sleep(delay)

    return _empty_row(arxiv_id, error=last_error)


def parse_paper(html: str, arxiv_id: str) -> dict[str, str]:
    """초록 페이지 HTML에서 제목, 저자, 초록 등을 추출한다."""
    soup = BeautifulSoup(html, "lxml")
    dateline = _text(soup.select_one("div.dateline"))
    submitted_match = _SUBMITTED_RE.search(dateline)

    author_block = soup.select_one("div.authors")
    author_names = (
        [anchor.get_text(strip=True) for anchor in author_block.select("a")]
        if author_block
        else []
    )

    return {
        "arxiv_id": arxiv_id,
        "title": _text_after_label(soup.select_one("h1.title"), "Title"),
        "authors": ", ".join(author_names) or _text_after_label(author_block, "Authors"),
        "abstract": _text_after_label(soup.select_one("blockquote.abstract"), "Abstract"),
        "subjects": _text(soup.select_one("td.tablecell.subjects")),
        "submitted": submitted_match.group(1).strip() if submitted_match else "",
        "comments": _text_after_label(
            soup.select_one("td.tablecell.comments"),
            "Comments",
        ),
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "error": "",
    }


def main(
    list_url: str = DEFAULT_LIST_URL,
    output_path: str | Path = DEFAULT_OUTPUT,
    delay: float = CRAWL_DELAY_SECONDS,
) -> pd.DataFrame:
    """최신 논문 ID를 수집하고 초록 페이지를 크롤링한 뒤 CSV로 저장한다."""
    arxiv_ids = collect_arxiv_ids(list_url)
    print(f"collected {len(arxiv_ids)} ids from {list_url}")
    if arxiv_ids and delay > 0:
        time.sleep(delay)

    papers = crawl_papers(arxiv_ids, delay=delay)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    papers.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved {len(papers)} rows to {output}")
    return papers


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


def _empty_row(arxiv_id: str, error: str) -> dict[str, str]:
    row = {column: "" for column in COLUMNS}
    row["arxiv_id"] = arxiv_id
    row["abs_url"] = f"https://arxiv.org/abs/{arxiv_id}"
    row["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}"
    row["error"] = error
    return row


if __name__ == "__main__":
    main()
