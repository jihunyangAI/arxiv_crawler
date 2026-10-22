"""paper.csv의 모든 논문에 한국어 쉬운 요약을 붙여 저장한다."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from summarize_paper import PAPER_CSV, summarize_paper

OUTPUT_CSV = Path(__file__).resolve().parent / "papers_with_summary.csv"
REQUEST_INTERVAL_SECONDS = 2

# 요청한 CSV 컬럼 이름. 값은 요약 함수의 real_world, limitation에서 채운다.
SUMMARY_COLUMNS = {
    "summary_one_line": "summary_one_line",
    "summary_easy": "summary_easy",
    "read_world": "real_world",
    "limitations": "limitation",
}


def process_all_papers(
    input_path: str | Path = PAPER_CSV,
    output_path: str | Path = OUTPUT_CSV,
    delay: float = REQUEST_INTERVAL_SECONDS,
    on_status: Callable[..., None] | None = None,
) -> pd.DataFrame:
    """각 논문 제목과 초록을 요약해 papers_with_summary.csv로 저장한다."""
    papers = pd.read_csv(input_path, dtype=str)
    output = Path(output_path)
    _restore_saved_summaries(papers, output)

    total = len(papers)
    for position, index in enumerate(papers.index, start=1):
        arxiv_id = papers.at[index, "arxiv_id"]
        if str(papers.at[index, "summary_one_line"]).strip():
            _notify(on_status, f"[{position}/{total}] {arxiv_id} 이미 요약됨", position, total)
            continue
        title = papers.at[index, "title"]
        _notify(on_status, f"[{position}/{total}] {arxiv_id} 요약 시작", position - 1, total)
        try:
            summary = summarize_paper(str(title), str(papers.at[index, "abstract"]))
        except Exception as exc:
            _notify(on_status, f"[{position}/{total}] {arxiv_id} 실패: {exc}", position, total)
        else:
            for column, source in SUMMARY_COLUMNS.items():
                papers.at[index, column] = summary[source]
            _notify(
                on_status,
                f"[{position}/{total}] {arxiv_id} 완료: {summary['summary_one_line']}",
                position,
                total,
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        papers.to_csv(output, index=False, encoding="utf-8-sig")
        if position < total and delay > 0:
            _notify(on_status, f"API 제한을 피해 {int(delay)}초 대기", position, total)
            time.sleep(delay)

    filled = int(papers["summary_one_line"].fillna("").ne("").sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    papers.to_csv(output, index=False, encoding="utf-8-sig")
    _notify(on_status, f"저장 {filled}/{total}건, 위치 {output}", total, total)
    return papers


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


def _restore_saved_summaries(papers: pd.DataFrame, output_path: Path) -> None:
    """이미 저장된 요약을 다시 받아 빈 행만 이어서 처리한다."""
    for column in SUMMARY_COLUMNS:
        papers[column] = ""
    if not output_path.is_file():
        return

    saved = pd.read_csv(output_path, dtype=str)
    if "arxiv_id" not in saved.columns or "summary_one_line" not in saved.columns:
        return

    saved = saved.drop_duplicates("arxiv_id").set_index("arxiv_id")
    for index in papers.index:
        arxiv_id = papers.at[index, "arxiv_id"]
        if arxiv_id not in saved.index:
            continue
        one_line = saved.at[arxiv_id, "summary_one_line"]
        if not isinstance(one_line, str) or not one_line.strip():
            continue
        for column in SUMMARY_COLUMNS:
            value = saved.at[arxiv_id, column] if column in saved.columns else ""
            papers.at[index, column] = "" if not isinstance(value, str) else value


if __name__ == "__main__":
    process_all_papers()
