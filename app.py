"""수집, 조회, 요약, 통계를 한 화면에서 쓰는 arXiv 도우미."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from collect_ids import (
    DEFAULT_LIST_URL,
    PAPER_CSV,
    extract_recent_papers,
    save_paper_csv,
)
from process_all_papers import OUTPUT_CSV, SUMMARY_COLUMNS, process_all_papers
from summarize_paper import summarize_paper

BASE_DIR = Path(__file__).resolve().parent

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "been", "being", "have", "has", "had", "not", "but", "its", "our", "their",
    "they", "them", "these", "those", "into", "over", "under", "between", "about",
    "after", "before", "than", "then", "also", "can", "may", "using", "used",
    "use", "via", "such", "which", "while", "where", "when", "what", "who",
    "how", "why", "each", "both", "more", "most", "other", "only", "paper",
    "papers", "approach", "approaches", "method", "methods", "based", "show",
    "shows", "shown", "results", "result", "propose", "proposed", "present",
    "across", "within", "without", "among", "through", "toward", "towards",
    "including", "new", "two", "three", "one", "all", "any", "will", "would",
    "could", "should", "does", "did",
}


def main() -> None:
    st.set_page_config(page_title="arXiv 논문 도우미", page_icon="📄", layout="wide")
    st.title("arXiv 논문 도우미")
    st.caption("최신 논문을 모으고, 쉬운 한국어로 요약한 뒤, 간단한 통계를 봅니다.")

    page = st.sidebar.radio(
        "메뉴",
        ["논문 수집", "저장된 논문", "쉬운 요약", "통계"],
        label_visibility="collapsed",
    )
    saved_count = _saved_count()
    st.sidebar.metric("저장된 논문", f"{saved_count}편")
    st.sidebar.caption("수집은 논문 1편당 약 15초, 요약은 다음 요청 전 2초를 기다립니다.")

    if page == "논문 수집":
        render_collect()
    elif page == "저장된 논문":
        render_library()
    elif page == "쉬운 요약":
        render_summary()
    else:
        render_stats()


def render_collect() -> None:
    st.header("논문 수집")
    st.write("arXiv 최신 목록에서 논문 정보를 가져와 `paper.csv`에 더합니다. 이미 있는 ID는 건너뜁니다.")

    left, right = st.columns(2)
    with left:
        category = st.text_input("카테고리", value="cs.AI", help="예: cs.AI, cs.CL. 전체 URL도 넣을 수 있습니다.")
    with right:
        limit = st.number_input("가져올 최신 논문 수", min_value=1, max_value=50, value=10, step=1)

    list_url = _list_url(category)
    st.caption(f"목록 주소: {list_url}")

    if st.button("수집 시작", type="primary"):
        on_status, progress = _bind_progress()
        with st.spinner("수집하는 중입니다."):
            papers = extract_recent_papers(limit=int(limit), list_url=list_url, on_status=on_status)
            saved_path = save_paper_csv(papers)
        progress.progress(100)
        if papers:
            st.success(f"새 논문 {len(papers)}편을 저장했습니다. {saved_path.name}")
            st.dataframe(_papers_frame(papers), width="stretch", hide_index=True)
        else:
            st.success("새로 받을 논문이 없습니다. 목록의 논문은 이미 저장되어 있습니다.")


def render_library() -> None:
    st.header("저장된 논문")
    papers = _load_papers()
    if papers.empty:
        st.info("아직 저장된 논문이 없습니다. 논문 수집에서 먼저 받아 주세요.")
        return

    query = st.text_input("제목 또는 ID 검색")
    view = papers
    if query.strip():
        mask = papers["title"].fillna("").str.contains(query, case=False) | papers["arxiv_id"].fillna("").str.contains(query)
        view = papers[mask]
    st.write(f"{len(view)}편")
    show_columns = [column for column in ["arxiv_id", "title", "authors", "submitted_date", "summary_one_line"] if column in view.columns]
    st.dataframe(view[show_columns], width="stretch", hide_index=True)

    if view.empty:
        return
    labels = [f"{row.arxiv_id} · {row.title}" for row in view.itertuples(index=False)]
    selected = st.selectbox("자세히 볼 논문", labels)
    arxiv_id = selected.split(" · ", 1)[0]
    paper = view[view["arxiv_id"] == arxiv_id].iloc[0]
    st.subheader(paper.get("title", ""))
    st.write(paper.get("authors", ""))
    meta = " · ".join(
        part
        for part in [str(paper.get("submitted_date", "")), str(paper.get("categories", ""))]
        if part and part != "nan"
    )
    if meta:
        st.caption(meta)
    if paper.get("pdf_url"):
        st.link_button("PDF 열기", paper["pdf_url"])
    st.markdown("**초록**")
    st.write(paper.get("abstract", ""))
    _show_summary_fields(paper)


def render_summary() -> None:
    st.header("쉬운 요약")
    st.write("Gemini로 한 줄 요약, 쉬운 설명, 현실 적용, 한계점을 만듭니다. 이미 요약된 논문은 다시 부르지 않습니다.")
    papers = _load_papers()
    if papers.empty:
        st.info("요약할 논문이 없습니다. 먼저 논문을 수집해 주세요.")
        return

    missing = _missing_summary_count(papers)
    st.caption(f"요약이 없는 논문 {missing}편")

    if st.button("빠진 요약 채우기", type="primary", disabled=missing == 0):
        on_status, progress = _bind_progress()
        with st.spinner("요약하는 중입니다."):
            result = process_all_papers(on_status=on_status)
        progress.progress(100)
        filled = int(result["summary_one_line"].fillna("").str.strip().ne("").sum())
        st.success(f"요약 {filled}편을 {OUTPUT_CSV.name}에 저장했습니다.")

    labels = [f"{row.arxiv_id} · {row.title}" for row in papers.itertuples(index=False)]
    selected = st.selectbox("논문 하나 요약", labels)
    if st.button("선택한 논문 요약"):
        arxiv_id = selected.split(" · ", 1)[0]
        paper = papers[papers["arxiv_id"] == arxiv_id].iloc[0]
        with st.spinner("이 논문을 요약하는 중입니다."):
            summary = summarize_paper(str(paper["title"]), str(paper["abstract"]))
            _save_one_summary(papers, arxiv_id, summary)
        st.success(summary["summary_one_line"])
        _show_summary_fields(pd.Series({
            "summary_one_line": summary["summary_one_line"],
            "summary_easy": summary["summary_easy"],
            "read_world": summary["real_world"],
            "limitations": summary["limitation"],
        }))


def render_stats() -> None:
    st.header("통계")
    papers = _load_papers()
    if papers.empty:
        st.info("통계를 낼 논문이 없습니다.")
        return

    stats = _paper_stats(papers)
    total, average, keyword = st.columns(3)
    total.metric("총 논문 수", f"{stats['total']}편")
    average.metric("저자 수 평균", f"{stats['avg_authors']:.1f}명")
    keyword.metric("가장 많이 나온 키워드", stats["top_keyword"], f"{stats['top_count']}회")
    st.caption("키워드는 제목과 초록에서 세었습니다. the, and 같은 기능어는 제외했습니다.")
    if stats["keywords"]:
        chart = pd.DataFrame(stats["keywords"], columns=["키워드", "횟수"]).set_index("키워드")
        st.bar_chart(chart)


def _list_url(category: str) -> str:
    text = category.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if not text:
        return DEFAULT_LIST_URL
    return f"https://arxiv.org/list/{text}/recent"


def _load_papers() -> pd.DataFrame:
    path = OUTPUT_CSV if OUTPUT_CSV.is_file() else PAPER_CSV
    if not path.is_file():
        return pd.DataFrame()
    papers = pd.read_csv(path, dtype=str).fillna("")
    for column in SUMMARY_COLUMNS:
        if column not in papers.columns:
            papers[column] = ""
    return papers


def _saved_count() -> int:
    papers = _load_papers()
    return 0 if papers.empty else len(papers)


def _missing_summary_count(papers: pd.DataFrame) -> int:
    if "summary_one_line" not in papers.columns:
        return len(papers)
    return int(papers["summary_one_line"].fillna("").str.strip().eq("").sum())


def _papers_frame(papers: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(papers)
    if "authors" in frame.columns:
        frame["authors"] = frame["authors"].apply(
            lambda value: ", ".join(value) if isinstance(value, list) else value
        )
    columns = [column for column in ["arxiv_id", "title", "authors", "submitted_date", "categories"] if column in frame.columns]
    return frame[columns]


def _save_one_summary(papers: pd.DataFrame, arxiv_id: str, summary: dict[str, str]) -> None:
    updated = papers.copy()
    for column in SUMMARY_COLUMNS:
        if column not in updated.columns:
            updated[column] = ""
    mask = updated["arxiv_id"] == arxiv_id
    for column, source in SUMMARY_COLUMNS.items():
        updated.loc[mask, column] = summary[source]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


def _show_summary_fields(paper: pd.Series) -> None:
    one_line = str(paper.get("summary_one_line", "")).strip()
    if not one_line:
        st.info("아직 요약이 없습니다.")
        return
    st.markdown(f"**한 줄 요약** {one_line}")
    easy = str(paper.get("summary_easy", "")).strip()
    if easy:
        st.markdown("**쉬운 설명**")
        st.write(easy)
    real_world = str(paper.get("read_world", "")).strip()
    if real_world:
        st.markdown("**어디에 쓰이나**")
        st.write(real_world)
    limitation = str(paper.get("limitations", "")).strip()
    if limitation:
        st.markdown("**한계**")
        st.write(limitation)


def _paper_stats(papers: pd.DataFrame) -> dict:
    author_counts = papers["authors"].map(_author_count) if "authors" in papers.columns else pd.Series(dtype=float)
    average = float(author_counts.mean()) if len(author_counts) else 0.0
    counts = Counter()
    pattern = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
    titles = papers["title"] if "title" in papers.columns else pd.Series("", index=papers.index)
    abstracts = papers["abstract"] if "abstract" in papers.columns else pd.Series("", index=papers.index)
    for text in (titles.fillna("") + " " + abstracts.fillna("")):
        tokens = [token.lower() for token in pattern.findall(str(text))]
        counts.update(token for token in tokens if token not in STOPWORDS)
    top = counts.most_common(8)
    keyword, frequency = top[0] if top else ("-", 0)
    return {
        "total": len(papers),
        "avg_authors": average,
        "top_keyword": keyword,
        "top_count": frequency,
        "keywords": top,
    }


def _author_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len([part for part in value.split(",") if part.strip()])


def _bind_progress():
    progress = st.progress(0)
    status = st.empty()
    log = st.empty()
    lines: list[str] = []

    def on_status(message: str, step: int | None = None, total: int | None = None) -> None:
        lines.append(message)
        status.info(message)
        log.code("\n".join(lines[-8:]))
        if step is not None and total:
            progress.progress(min(int(step / total * 100), 100))

    return on_status, progress


if __name__ == "__main__":
    main()
