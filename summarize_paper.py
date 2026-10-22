"""Gemini API로 논문 제목과 초록을 쉬운 한국어 요약으로 바꾼다."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

# 2026-10-22 공식 가격표의 표준 요금이 싼 순서다.
# 2.5 Flash-Lite는 입력 $0.10 / 출력 $0.40로 가장 싸지만, 신규 키는 호출이 막힌다.
# 그다음이 3.1 Flash-Lite($0.25 / $1.50), 이어서 3.5 Flash-Lite($0.30 / $2.50)다.
# https://ai.google.dev/gemini-api/docs/pricing
MODEL_CANDIDATES = (
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
)
used_model = ""
_skipped_models: set[str] = set()
ENV_PATH = Path(__file__).resolve().parent / ".env.local"
PAPER_CSV = Path(__file__).resolve().parent / "paper.csv"

SYSTEM_PROMPT = """당신은 논문 요약기다. 입력으로 주어진 제목과 초록만 사용하고, 없는 사실은 만들지 마라.
반드시 한국어 JSON 객체 하나만 출력한다. 키는 아래 4개뿐이며 이름을 바꾸거나 빼지 마라.
마크다운, 코드블록, 설명 문장은 붙이지 마라.

summary_one_line:
- 논문이 무엇을 하는지 한 줄로 말한다.
- 공백을 포함해 20자 이내다.
- 마침표, 느낌표, 물음표, 줄바꿈을 넣지 않는다.

summary_easy:
- 중학생이 읽을 수 있게 쉬운 말로 설명한다.
- 문장은 3개 또는 4개다. 그보다 적거나 많지 않다.
- 각 문장은 마침표로 끝낸다.
- 전문용어는 바로 옆에서 풀어 쓴다.
- 줄바꿈을 넣지 않는다.

real_world:
- 이 연구가 실제로 어디에 쓰이는지 말한다.
- 2줄 또는 3줄이다. 각 줄은 하나의 사용 장면이다.
- 줄과 줄 사이는 줄바꿈 문자 하나만 넣는다.
- 각 줄 끝에 마침표를 찍는다.

limitation:
- 이 논문의 한계나 아직 확인되지 않은 점을 말한다.
- 1줄 또는 2줄이다.
- 줄과 줄 사이는 줄바꿈 문자 하나만 넣는다.
- 각 줄 끝에 마침표를 찍는다.
"""


class PaperSummary(BaseModel):
    summary_one_line: str = Field(
        description="공백 포함 20자 이내 한국어 한 줄 요약. 마침표와 줄바꿈 없음."
    )
    summary_easy: str = Field(
        description="중학생도 이해하는 한국어 설명. 마침표로 끝나는 문장 3개 또는 4개. 줄바꿈 없음."
    )
    real_world: str = Field(
        description="현실 적용. 마침표로 끝나는 2줄 또는 3줄. 줄은 줄바꿈 하나로 구분."
    )
    limitation: str = Field(
        description="한계점. 마침표로 끝나는 1줄 또는 2줄. 줄은 줄바꿈 하나로 구분."
    )


def summarize_paper(title: str, abstract: str) -> dict[str, str]:
    """논문 제목과 초록을 받아 고정된 4개 항목의 요약 딕셔너리를 반환한다."""
    client = genai.Client(
        api_key=_load_api_key(),
        http_options=types.HttpOptions(timeout=45_000),
    )
    prompt = _user_prompt(title, abstract)
    model, summary = _generate_with_fallback(client, prompt)
    summary = _normalize_summary(summary)
    problems = _format_problems(summary)
    if problems:
        summary = _normalize_summary(
            _generate_summary(
                client,
                model,
                _repair_prompt(prompt, summary, problems),
            )
        )
        problems = _format_problems(summary)
        if problems:
            raise ValueError("요약 형식이 맞지 않습니다: " + "; ".join(problems))
    return summary.model_dump()


def _generate_with_fallback(
    client: genai.Client,
    prompt: str,
) -> tuple[str, PaperSummary]:
    """가격이 싼 모델부터 시도하고, 신규 키에서 막힌 모델은 건너뛴다."""
    global used_model
    last_error: Exception | None = None
    candidates = [model for model in MODEL_CANDIDATES if model not in _skipped_models]
    if used_model and used_model in candidates:
        candidates = [used_model, *[model for model in candidates if model != used_model]]
    for model in candidates:
        for attempt in range(5):
            try:
                summary = _generate_summary(client, model, prompt)
                used_model = model
                return model, summary
            except errors.ClientError as exc:
                if getattr(exc, "code", None) == 404:
                    _skipped_models.add(model)
                    last_error = exc
                    break
                raise
            except errors.ServerError as exc:
                last_error = exc
                if getattr(exc, "code", None) != 503 or attempt == 4:
                    break
                time.sleep(5 * (attempt + 1))
    raise RuntimeError("사용 가능한 Gemini 모델이 없습니다.") from last_error


def _generate_summary(client: genai.Client, model: str, prompt: str) -> PaperSummary:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=_request_config(model),
    )
    if isinstance(response.parsed, PaperSummary):
        return response.parsed
    return PaperSummary.model_validate_json(response.text or "")


def _request_config(model: str) -> types.GenerateContentConfig:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        max_output_tokens=1024,
        response_mime_type="application/json",
        response_schema=PaperSummary,
    )
    # 3.5부터 thinking_budget은 거부된다. 그 이전 모델은 사고를 끄면 출력 비용이 줄어든다.
    if model.startswith("gemini-2.") or "3.1-flash-lite" in model:
        config.thinking_config = types.ThinkingConfig(thinking_budget=0)
    return config


def _user_prompt(title: str, abstract: str) -> str:
    return f"제목:\n{title.strip()}\n\n초록:\n{abstract.strip()}"


def _repair_prompt(prompt: str, summary: PaperSummary, problems: list[str]) -> str:
    current = json.dumps(summary.model_dump(), ensure_ascii=False)
    issues = "\n".join(f"- {problem}" for problem in problems)
    return (
        f"{prompt}\n\n"
        "이전 출력은 형식이 맞지 않는다. 같은 4개 키로 다시 작성하라.\n"
        f"이전 출력:\n{current}\n"
        f"고칠 점:\n{issues}"
    )


def _normalize_summary(summary: PaperSummary) -> PaperSummary:
    """한 줄로 붙은 적용·한계 문장을 요청한 줄 수로 나눈다."""
    one_line = summary.summary_one_line.replace("\n", " ").strip().rstrip(".!?。")
    easy = " ".join(part.strip() for part in summary.summary_easy.splitlines() if part.strip())
    return PaperSummary(
        summary_one_line=one_line,
        summary_easy=easy,
        real_world=_as_lines(summary.real_world, 2, 3),
        limitation=_as_lines(summary.limitation, 1, 2),
    )


def _as_lines(text: str, minimum: int, maximum: int) -> str:
    lines = _lines(text)
    if minimum <= len(lines) <= maximum:
        return "\n".join(lines)
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?。])\s+", text.replace("\n", " ").strip())
        if part.strip()
    ]
    if minimum <= len(sentences) <= maximum:
        return "\n".join(sentences)
    return text.strip()


def _format_problems(summary: PaperSummary) -> list[str]:
    problems: list[str] = []
    one_line = summary.summary_one_line.strip()
    if not one_line:
        problems.append("summary_one_line이 비어 있다")
    if len(one_line) > 20:
        problems.append(f"summary_one_line이 {len(one_line)}자라서 20자를 넘는다")
    if re.search(r"[.!?。\n]", one_line):
        problems.append("summary_one_line에 마침표나 줄바꿈이 있다")

    easy_count = _sentence_count(summary.summary_easy)
    if easy_count not in {3, 4}:
        problems.append(f"summary_easy 문장 수가 {easy_count}개다. 3개 또는 4개여야 한다")
    if "\n" in summary.summary_easy:
        problems.append("summary_easy에 줄바꿈이 있다")

    real_world_lines = _lines(summary.real_world)
    if len(real_world_lines) not in {2, 3}:
        problems.append(f"real_world 줄 수가 {len(real_world_lines)}줄이다. 2줄 또는 3줄이어야 한다")

    limitation_lines = _lines(summary.limitation)
    if len(limitation_lines) not in {1, 2}:
        problems.append(f"limitation 줄 수가 {len(limitation_lines)}줄이다. 1줄 또는 2줄이어야 한다")
    return problems


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。])\s+", text.strip()) if part.strip()]
    return len(parts)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _load_api_key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    if not ENV_PATH.is_file():
        raise FileNotFoundError(f"API 키 파일이 없습니다: {ENV_PATH}")

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == "GEMINI_API_KEY":
            api_key = value.strip().strip('"').strip("'")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                return api_key
    raise RuntimeError("GEMINI_API_KEY가 .env.local에 없습니다.")


if __name__ == "__main__":
    first_paper = pd.read_csv(PAPER_CSV, dtype=str).iloc[0]
    result = summarize_paper(first_paper["title"], first_paper["abstract"])
    print(f"model={used_model}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
