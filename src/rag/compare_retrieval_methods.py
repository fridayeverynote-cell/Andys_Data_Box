# ============================================================
# - BM25 / Dense / RRF 검색 비교
# - 비교 결과 CSV 저장
# - 수동 평가 점수 집계
# - 시각화 저장
# - 실패 사례 CSV 저장
# - 덮어쓰기 방지
# ============================================================

from pathlib import Path
from collections import defaultdict
import os

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAG_TEXT_PATH = PROCESSED_DATA_DIR / "rag_documents_with_text.csv"
VECTOR_DB_DIR = PROCESSED_DATA_DIR / "faiss_rag_db"

COMPARE_OUTPUT_PATH = PROCESSED_DATA_DIR / "retrieval_compare_results.csv"
SUMMARY_OUTPUT_PATH = PROCESSED_DATA_DIR / "retrieval_evaluation_summary.csv"
PLOT_OUTPUT_PATH = PROCESSED_DATA_DIR / "retrieval_method_scores.png"
FAILURE_CASES_OUTPUT_PATH = PROCESSED_DATA_DIR / "retrieval_failure_cases.csv"


# ============================================================
# 2. 환경 변수 로드
# ============================================================
def load_api_key() -> str:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return openai_api_key


# ============================================================
# 3. 공통 유틸
# ============================================================
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_test_queries() -> list[str]:
    return [
        "남자친구가 내 말을 대충 듣는 것 같아서 서운해. 어떻게 말하면 좋을까?",
        "내가 힘들다고 했는데 공감 없이 넘어가서 속상해.",
        "상대가 자꾸 장난처럼 넘겨서 진지하게 대화하고 싶어.",
        "다투고 나서 어색한데 먼저 뭐라고 보내야 할지 모르겠어.",
        "내 감정을 몰라주는 것 같아서 답답하고 화가 나.",
        "답장이 늦고 성의가 없어서 무시당하는 느낌이 들어.",
        "사소한 말다툼이 반복돼서 지쳐.",
    ]


# ============================================================
# 4. 데이터 로드
# ============================================================
def load_rag_dataframe() -> pd.DataFrame:
    if not RAG_TEXT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {RAG_TEXT_PATH}")

    rag_df = pd.read_csv(RAG_TEXT_PATH)

    if "rag_text" not in rag_df.columns:
        raise ValueError("rag_text 컬럼이 없습니다.")

    rag_df["rag_text"] = rag_df["rag_text"].astype(str).str.strip()
    rag_df = rag_df[rag_df["rag_text"] != ""].copy()
    rag_df = rag_df.reset_index(drop=True)

    return rag_df


def build_bm25(rag_df: pd.DataFrame) -> BM25Okapi:
    documents = rag_df["rag_text"].tolist()
    tokenized_docs = [doc.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


def load_vector_db(openai_api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(f"벡터DB 폴더가 없습니다: {VECTOR_DB_DIR}")

    vector_db = FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_db


# ============================================================
# 5. 검색 함수
# ============================================================
def bm25_search(query: str, rag_df: pd.DataFrame, bm25: BM25Okapi, k: int = 3) -> list[dict]:
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    result_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []
    for rank, idx in enumerate(result_indices, start=1):
        row = rag_df.iloc[idx]
        results.append({
            "method": "bm25",
            "rank": rank,
            "dialogue_id": clean_text(row.get("dialogue_id", "")),
            "situation": clean_text(row.get("situation", "")),
            "speaker_emotion": clean_text(row.get("speaker_emotion", "")),
            "risk_level": clean_text(row.get("risk_level", "")),
            "score": float(scores[idx]),
            "page_content_preview": clean_text(row["rag_text"])[:300],
        })
    return results


def dense_search(query: str, vector_db: FAISS, k: int = 3) -> list[dict]:
    docs_and_scores = vector_db.similarity_search_with_score(query, k=k)

    results = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        results.append({
            "method": "dense",
            "rank": rank,
            "dialogue_id": clean_text(doc.metadata.get("dialogue_id", "")),
            "situation": clean_text(doc.metadata.get("situation", "")),
            "speaker_emotion": clean_text(doc.metadata.get("speaker_emotion", "")),
            "risk_level": clean_text(doc.metadata.get("risk_level", "")),
            "score": float(score),
            "page_content_preview": clean_text(doc.page_content)[:300],
        })
    return results


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_n: int = 3) -> list[dict]:
    fused_scores = defaultdict(float)
    item_lookup = {}

    for result_list in result_lists:
        for item in result_list:
            dialogue_id = item["dialogue_id"]
            rank = item["rank"]

            if not dialogue_id:
                continue

            fused_scores[dialogue_id] += 1 / (k + rank)
            item_lookup[dialogue_id] = item

    ranked_ids = sorted(
        fused_scores.keys(),
        key=lambda x: fused_scores[x],
        reverse=True
    )[:top_n]

    fused_results = []
    for rank, dialogue_id in enumerate(ranked_ids, start=1):
        base_item = item_lookup[dialogue_id].copy()
        base_item["method"] = "rrf"
        base_item["rank"] = rank
        base_item["score"] = fused_scores[dialogue_id]
        fused_results.append(base_item)

    return fused_results


# ============================================================
# 6. 비교 결과 생성
# ============================================================
def build_compare_results(
    rag_df: pd.DataFrame,
    bm25: BM25Okapi,
    vector_db: FAISS,
    queries: list[str],
    k: int = 3
) -> pd.DataFrame:
    all_rows = []

    for query in queries:
        bm25_results = bm25_search(query, rag_df=rag_df, bm25=bm25, k=k)
        dense_results = dense_search(query, vector_db=vector_db, k=k)
        rrf_results = reciprocal_rank_fusion([bm25_results, dense_results], top_n=k)

        for row in bm25_results + dense_results + rrf_results:
            row["query"] = query
            row["is_relevant"] = ""
            row["emotion_match"] = ""
            row["usable_for_reply"] = ""
            row["failure_type"] = ""
            row["failure_reason"] = ""
            all_rows.append(row)

    result_df = pd.DataFrame(all_rows)
    result_df = result_df[
        [
            "query", "method", "rank", "dialogue_id", "situation",
            "speaker_emotion", "risk_level", "score",
            "page_content_preview", "is_relevant",
            "emotion_match", "usable_for_reply",
            "failure_type", "failure_reason"
        ]
    ]

    result_df.to_csv(COMPARE_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {COMPARE_OUTPUT_PATH}")

    return result_df


# ============================================================
# 7. 평가 결과 요약
# ============================================================
def summarize_evaluation() -> pd.DataFrame | None:
    if not COMPARE_OUTPUT_PATH.exists():
        print("retrieval_compare_results.csv가 아직 없습니다.")
        return None

    eval_df = pd.read_csv(COMPARE_OUTPUT_PATH)
    score_cols = ["is_relevant", "emotion_match", "usable_for_reply"]

    for col in score_cols:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    valid_df = eval_df.dropna(subset=score_cols).copy()

    if valid_df.empty:
        print("평가 점수가 아직 입력되지 않았습니다.")
        return None

    summary_df = valid_df.groupby("method")[score_cols].mean().reset_index()
    summary_df["overall_mean"] = summary_df[score_cols].mean(axis=1)

    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {SUMMARY_OUTPUT_PATH}")
    print(summary_df)

    return summary_df


# ============================================================
# 8. 실패 사례 저장
# ============================================================
def save_failure_cases(top_n: int = 10) -> pd.DataFrame | None:
    if not COMPARE_OUTPUT_PATH.exists():
        print("retrieval_compare_results.csv가 아직 없습니다.")
        return None

    eval_df = pd.read_csv(COMPARE_OUTPUT_PATH)
    score_cols = ["is_relevant", "emotion_match", "usable_for_reply"]

    for col in score_cols:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    valid_df = eval_df.dropna(subset=score_cols).copy()

    if valid_df.empty:
        print("평가 점수가 아직 입력되지 않았습니다.")
        return None

    valid_df["failure_score"] = valid_df[score_cols].sum(axis=1)

    failure_df = valid_df.sort_values(
        by=["failure_score", "method", "rank"],
        ascending=[True, True, True]
    ).head(top_n)

    failure_df.to_csv(FAILURE_CASES_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {FAILURE_CASES_OUTPUT_PATH}")

    return failure_df


# ============================================================
# 9. 시각화
# ============================================================
def plot_summary(summary_df: pd.DataFrame | None) -> None:
    if summary_df is None or summary_df.empty:
        print("시각화할 summary_df가 없습니다.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["method"], summary_df["overall_mean"])
    plt.xlabel("Retrieval Method")
    plt.ylabel("Average Score")
    plt.title("Average Retrieval Evaluation Score by Method")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"[저장 완료] {PLOT_OUTPUT_PATH}")


# ============================================================
# 10. 메인 실행
# - 첫 실행: 비교용 CSV 생성만 수행
# - CSV에 0/1 및 실패 분석 입력 후 재실행: 요약 + 그래프 + 실패 사례 저장
# ============================================================
def main() -> None:
    openai_api_key = load_api_key()
    rag_df = load_rag_dataframe()
    bm25 = build_bm25(rag_df)
    vector_db = load_vector_db(openai_api_key)
    queries = get_test_queries()

    if not COMPARE_OUTPUT_PATH.exists():
        build_compare_results(
            rag_df=rag_df,
            bm25=bm25,
            vector_db=vector_db,
            queries=queries,
            k=3,
        )
        print("\n비교용 CSV를 생성했습니다.")
        print("retrieval_compare_results.csv에 0/1 및 failure_type, failure_reason을 입력한 뒤 다시 실행하세요.")
        return

    print("기존 retrieval_compare_results.csv를 사용합니다. 덮어쓰지 않습니다.")
    summary_df = summarize_evaluation()
    plot_summary(summary_df)
    save_failure_cases(top_n=10)


if __name__ == "__main__":
    main()