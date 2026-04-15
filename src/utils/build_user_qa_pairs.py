"""build_user_qa_pairs  –  response_pairs + rag_documents → user_qa_pairs.csv

PS1 버전의 로직을 메인으로 삼아 Python 으로 재작성한 스크립트.
행 단위 순회 / dict 카운터 / 명시적 기본값 처리를 충실히 반영한다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


# ── 기본 경로 ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_EMPATHY = "미분류"


# ── 헬퍼 함수 ────────────────────────────────────────────────

def convert_context_roles(context: str) -> str:
    """speaker: / listener: 접두어를 user: / assistant: 로 치환한다.

    PS1 ``Convert-ContextRoles`` 와 동일한 로직.
    """
    lines: list[str] = []

    for raw_line in str(context or "").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("speaker:"):
            lines.append("user: " + line[len("speaker:"):].strip())
        elif line.startswith("listener:"):
            lines.append("assistant: " + line[len("listener:"):].strip())
        else:
            lines.append(line)

    return "\n".join(lines)


def get_last_user_question(context: str) -> str:
    """컨텍스트에서 마지막 speaker 발화를 추출한다.

    PS1 ``Get-LastUserQuestion`` 와 동일한 로직.
    """
    for raw_line in reversed(str(context or "").splitlines()):
        line = raw_line.strip()

        if line.startswith("speaker:"):
            return line[len("speaker:"):].strip()

    return ""


def _safe(row: dict[str, Any], key: str, default: str = "") -> str:
    """row 에서 key 를 안전하게 꺼낸다. 없거나 빈 문자열이면 default 를 반환."""
    value = row.get(key, default)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


# ── 메인 빌드 ────────────────────────────────────────────────

def build(processed_dir: Path) -> None:
    response_pairs_path = processed_dir / "response_pairs.csv"
    rag_documents_path = processed_dir / "rag_documents.csv"
    output_path = processed_dir / "user_qa_pairs.csv"

    # ── response_pairs 로드 ──────────────────────────────────
    if not response_pairs_path.exists():
        raise FileNotFoundError(f"Missing source file: {response_pairs_path}")

    with open(response_pairs_path, encoding="utf-8-sig") as fh:
        response_rows = list(csv.DictReader(fh))

    # ── rag_documents 메타 로드 (첫 번째 행만, PS1 방식) ─────
    rag_metadata: dict[str, dict[str, Any]] = {}

    if rag_documents_path.exists():
        with open(rag_documents_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                dialogue_id = str(row.get("dialogue_id", ""))
                if dialogue_id not in rag_metadata:
                    rag_metadata[dialogue_id] = row

    # ── 행 단위 순회하며 QA 페어 구성 ───────────────────────
    dialogue_counters: dict[str, int] = {}
    output_rows: list[dict[str, Any]] = []

    for row in response_rows:
        dialogue_id = str(row.get("dialogue_id", ""))

        dialogue_counters[dialogue_id] = dialogue_counters.get(dialogue_id, 0) + 1
        response_index = dialogue_counters[dialogue_id]

        meta = rag_metadata.get(dialogue_id)

        # 기본값 – PS1 과 동일
        risk_level = "unknown"
        grade = ""
        avg_rating = ""
        final_speaker_change_emotion = "unknown"

        if meta is not None:
            risk_level = _safe(meta, "risk_level", "unknown")
            grade = _safe(meta, "grade", "")
            avg_rating = _safe(meta, "avg_rating", "")
            final_speaker_change_emotion = _safe(
                meta, "final_speaker_change_emotion", "unknown"
            )

        terminate_text = str(row.get("terminate", "")).strip().lower()
        is_terminal = terminate_text in ("true", "1", "yes")

        listener_empathy = str(row.get("listener_empathy", "")).strip()

        output_rows.append(
            {
                "qa_pair_id": f"{dialogue_id}_R{response_index:02d}",
                "dialogue_id": dialogue_id,
                "response_index": response_index,
                "relation": _safe(row, "relation"),
                "situation": _safe(row, "situation"),
                "user_emotion": _safe(row, "speaker_emotion"),
                "risk_level": risk_level,
                "user_context": convert_context_roles(
                    row.get("context_before_response", "")
                ),
                "user_question": get_last_user_question(
                    row.get("context_before_response", "")
                ),
                "assistant_answer": str(row.get("listener_response", "")).strip(),
                "answer_empathy": listener_empathy if listener_empathy else DEFAULT_EMPATHY,
                "is_terminal": is_terminal,
                "grade": grade,
                "avg_rating": avg_rating,
                "final_speaker_change_emotion": final_speaker_change_emotion,
            }
        )

    # ── CSV 저장 ─────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "qa_pair_id",
        "dialogue_id",
        "response_index",
        "relation",
        "situation",
        "user_emotion",
        "risk_level",
        "user_context",
        "user_question",
        "assistant_answer",
        "answer_empathy",
        "is_terminal",
        "grade",
        "avg_rating",
        "final_speaker_change_emotion",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"[SAVE] {output_path}")

    # ── 요약 출력 ────────────────────────────────────────────
    _print_summary(output_rows, dialogue_counters)


# ── 요약 출력 ────────────────────────────────────────────────

def _print_summary(
    rows: list[dict[str, Any]],
    dialogue_counters: dict[str, int],
) -> None:
    empty_question = sum(1 for r in rows if r["user_question"] == "")
    empty_answer = sum(1 for r in rows if r["assistant_answer"] == "")

    print()
    print("=" * 60)
    print("[USER QA PAIRS SUMMARY]")
    print("=" * 60)
    print(f"qa pair rows: {len(rows)}")
    print(f"dialogues: {len(dialogue_counters)}")
    print(f"empty user_question: {empty_question}")
    print(f"empty assistant_answer: {empty_answer}")

    # 추가 분포 (기존 PY 버전에만 있던 항목 보존)
    _print_value_counts(rows, "user_emotion")
    _print_value_counts(rows, "risk_level")

    # 샘플 출력
    print("\n[sample]")
    for r in rows[:3]:
        print(r)


def _print_value_counts(rows: list[dict[str, Any]], key: str) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        val = r.get(key, "")
        counts[val] = counts.get(val, 0) + 1

    print(f"\n[{key}]")
    for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {val}: {cnt}")


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="response_pairs + rag_documents → user_qa_pairs.csv"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="processed 데이터 디렉터리 경로",
    )
    args = parser.parse_args()

    build(args.processed_dir)


if __name__ == "__main__":
    main()
