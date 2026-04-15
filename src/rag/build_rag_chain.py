# ============================================================
# - BM25 / Dense / RRF 기반 검색
# - response example 연결
# - LLM으로 추천 답변 생성
# - 상황 요약 / 감정 / 위험도 / 추천 답변 / 피해야 할 표현 출력 강화
# ============================================================

from pathlib import Path
import os
from collections import Counter, defaultdict

import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAG_TEXT_PATH = PROCESSED_DATA_DIR / "rag_documents_with_text.csv"
RESPONSE_TEXT_PATH = PROCESSED_DATA_DIR / "response_pairs_with_text.csv"
VECTOR_DB_DIR = PROCESSED_DATA_DIR / "faiss_rag_db"


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
# 3. 유틸
# ============================================================
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def infer_emotion_from_question(question: str) -> str:
    q = clean_text(question)

    if any(kw in q for kw in ["서운", "속상", "상처"]):
        return "슬픔"
    if any(kw in q for kw in ["화", "짜증", "무시", "열받"]):
        return "분노"
    if any(kw in q for kw in ["불안", "걱정"]):
        return "불안"
    if any(kw in q for kw in ["답답", "지쳐"]):
        return "답답함"
    return ""


def extract_keywords_from_question(question: str) -> list[str]:
    candidates = [
        "서운", "속상", "무시", "안 들어", "대충", "화", "상처",
        "답답", "공감", "진지", "읽씹", "답장", "회피", "장난", "집중",
        "연락", "말다툼", "반복", "지쳐"
    ]
    return [kw for kw in candidates if kw in question]


# ============================================================
# 4. 데이터/모델 로드
# ============================================================
def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAG_TEXT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {RAG_TEXT_PATH}")
    if not RESPONSE_TEXT_PATH.exists():
        raise FileNotFoundError(f"파일이 없습니다: {RESPONSE_TEXT_PATH}")

    rag_df = pd.read_csv(RAG_TEXT_PATH)
    response_df = pd.read_csv(RESPONSE_TEXT_PATH)

    if "rag_text" not in rag_df.columns:
        raise ValueError("rag_df에 rag_text 컬럼이 없습니다.")

    rag_df["rag_text"] = rag_df["rag_text"].astype(str).str.strip()
    rag_df = rag_df[rag_df["rag_text"] != ""].copy()
    rag_df = rag_df.reset_index(drop=True)

    return rag_df, response_df


def build_bm25(rag_df: pd.DataFrame) -> BM25Okapi:
    docs = rag_df["rag_text"].tolist()
    tokenized_docs = [doc.split() for doc in docs]
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


def load_llm(openai_api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=openai_api_key,
    )


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
    for idx in result_indices:
        row = rag_df.iloc[idx]
        results.append({
            "dialogue_id": clean_text(row.get("dialogue_id", "")),
            "relation": clean_text(row.get("relation", "")),
            "situation": clean_text(row.get("situation", "")),
            "speaker_emotion": clean_text(row.get("speaker_emotion", "")),
            "risk_level": clean_text(row.get("risk_level", "")),
            "page_content": clean_text(row.get("rag_text", "")),
        })
    return results


def dense_search(query: str, vector_db: FAISS, k: int = 3) -> list[dict]:
    docs = vector_db.similarity_search(query, k=k)

    results = []
    for doc in docs:
        results.append({
            "dialogue_id": clean_text(doc.metadata.get("dialogue_id", "")),
            "relation": clean_text(doc.metadata.get("relation", "")),
            "situation": clean_text(doc.metadata.get("situation", "")),
            "speaker_emotion": clean_text(doc.metadata.get("speaker_emotion", "")),
            "risk_level": clean_text(doc.metadata.get("risk_level", "")),
            "page_content": clean_text(doc.page_content),
        })
    return results


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_n: int = 3) -> list[dict]:
    fused_scores = defaultdict(float)
    item_lookup = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, start=1):
            dialogue_id = item["dialogue_id"]
            if not dialogue_id:
                continue
            fused_scores[dialogue_id] += 1 / (k + rank)
            item_lookup[dialogue_id] = item

    ranked_ids = sorted(
        fused_scores.keys(),
        key=lambda x: fused_scores[x],
        reverse=True
    )[:top_n]

    return [item_lookup[dialogue_id] for dialogue_id in ranked_ids]


def retrieve_documents(
    question: str,
    rag_df: pd.DataFrame,
    bm25: BM25Okapi,
    vector_db: FAISS,
    method: str = "rrf",
    k: int = 3
) -> list[dict]:
    if method == "dense":
        return dense_search(question, vector_db=vector_db, k=k)

    if method == "bm25":
        return bm25_search(question, rag_df=rag_df, bm25=bm25, k=k)

    if method == "rrf":
        bm25_results = bm25_search(question, rag_df=rag_df, bm25=bm25, k=k)
        dense_results = dense_search(question, vector_db=vector_db, k=k)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_n=k)

    raise ValueError("method는 'dense', 'bm25', 'rrf' 중 하나여야 합니다.")


# ============================================================
# 6. 감정 / 위험도 / 상황 요약
# ============================================================
def get_main_emotion(question: str, retrieved_docs: list[dict]) -> str:
    question_emotion = infer_emotion_from_question(question)
    if question_emotion:
        return question_emotion

    emotions = [
        clean_text(doc.get("speaker_emotion", ""))
        for doc in retrieved_docs
        if clean_text(doc.get("speaker_emotion", ""))
    ]

    if not emotions:
        return "미상"

    return Counter(emotions).most_common(1)[0][0]


def get_main_risk_level(retrieved_docs: list[dict]) -> str:
    risk_levels = [
        clean_text(doc.get("risk_level", ""))
        for doc in retrieved_docs
        if clean_text(doc.get("risk_level", ""))
    ]

    if not risk_levels:
        return "미상"

    normalized = []
    risk_map = {
        "낮음": "낮음",
        "보통": "보통",
        "중간": "보통",
        "높음": "높음",
    }

    for risk in risk_levels:
        normalized.append(risk_map.get(risk, risk))

    counter = Counter(normalized)
    return counter.most_common(1)[0][0]


def summarize_current_situation(question: str, retrieved_docs: list[dict]) -> str:
    if not retrieved_docs:
        return clean_text(question)

    top_doc = retrieved_docs[0]
    relation = clean_text(top_doc.get("relation", "연인"))
    situation = clean_text(top_doc.get("situation", ""))
    emotion = clean_text(top_doc.get("speaker_emotion", ""))

    parts = []
    if relation:
        parts.append(f"{relation} 관계에서")
    if situation:
        parts.append(situation)
    if emotion:
        parts.append(f"주된 감정은 {emotion}")

    if parts:
        return ", ".join(parts)

    return clean_text(question)


# ============================================================
# 7. response example 연결
# ============================================================
def filter_response_examples_by_dialogue_ids(response_df: pd.DataFrame, dialogue_ids: list[str]) -> pd.DataFrame:
    if "dialogue_id" not in response_df.columns:
        return response_df.copy()

    return response_df[
        response_df["dialogue_id"].astype(str).isin(dialogue_ids)
    ].copy()


def score_response_example(row: pd.Series, emotion: str, question_keywords: list[str]) -> int:
    score = 0

    relation = clean_text(row.get("relation", ""))
    situation = clean_text(row.get("situation", ""))
    speaker_emotion = clean_text(row.get("speaker_emotion", ""))
    context_before_response = clean_text(row.get("context_before_response", ""))
    listener_response = clean_text(row.get("listener_response", ""))
    listener_empathy = clean_text(row.get("listener_empathy", ""))

    merged_text = " ".join([
        relation, situation, speaker_emotion,
        context_before_response, listener_response, listener_empathy
    ])

    if relation == "연인":
        score += 2
    if emotion and emotion in speaker_emotion:
        score += 3
    for kw in question_keywords:
        if kw in merged_text:
            score += 2
    if listener_empathy and listener_empathy != "미분류":
        score += 1

    return score


def get_response_examples(
    response_df: pd.DataFrame,
    retrieved_docs: list[dict],
    emotion: str,
    question: str,
    top_n: int = 3
) -> str:
    dialogue_ids = [
        clean_text(doc.get("dialogue_id", ""))
        for doc in retrieved_docs
        if clean_text(doc.get("dialogue_id", ""))
    ]

    candidate_df = filter_response_examples_by_dialogue_ids(response_df, dialogue_ids)

    if candidate_df.empty:
        candidate_df = response_df.copy()

    question_keywords = extract_keywords_from_question(question)

    candidate_df["score"] = candidate_df.apply(
        lambda row: score_response_example(row, emotion, question_keywords),
        axis=1
    )

    candidate_df = candidate_df.sort_values(by="score", ascending=False)

    text_col = "response_example_text" if "response_example_text" in candidate_df.columns else "listener_response"
    selected = candidate_df.head(top_n)[text_col].astype(str).tolist()

    return "\n\n".join(
        [f"[응답 예시 {i + 1}]\n{ex}" for i, ex in enumerate(selected)]
    )


def format_docs(docs: list[dict]) -> str:
    formatted = []

    for i, doc in enumerate(docs, start=1):
        block = [
            f"[유사 사례 {i}]",
            f"dialogue_id: {doc.get('dialogue_id', '')}",
            f"관계: {doc.get('relation', '')}",
            f"상황: {doc.get('situation', '')}",
            f"화자 감정: {doc.get('speaker_emotion', '')}",
            f"위험도: {doc.get('risk_level', '')}",
            f"본문 일부: {doc.get('page_content', '')[:500]}",
        ]
        formatted.append("\n".join(block))

    return "\n\n".join(formatted)


# ============================================================
# 8. 프롬프트
# ============================================================
PROMPT = PromptTemplate.from_template(
    """
너는 연인 갈등 상황에서 더 나은 답변을 추천하는 도우미다.

사용자가 연인과의 갈등 상황을 설명하면,
현재 상황 요약, 감정, 위험도, 추천 답변, 피해야 할 표현과 대체 표현을
이해하기 쉽게 정리해야 한다.

반드시 아래 조건을 지켜야 한다.
- 너무 길지 않게 작성할 것
- 공격적이거나 비난하는 말투는 피할 것
- 감정은 솔직하지만 부드럽게 전달할 것
- 실제 채팅에 바로 복붙해서 보낼 수 있는 문장으로 작성할 것
- 지나치게 상담사처럼 말하지 말 것
- 사용자의 현재 감정 톤을 반영할 것
- 유사 사례의 맥락을 참고하되, 현재 질문에 더 직접 맞게 작성할 것

사용자 입력:
{question}

상황 요약:
{situation_summary}

대표 감정:
{main_emotion}

갈등 위험도:
{risk_level}

유사 사례:
{context}

참고 응답 예시:
{response_examples}

아래 형식으로 출력할 것.

[상황 요약]
...

[감정]
...

[위험도]
...

[추천 답변 1]
...

[추천 답변 2]
...

[피해야 할 표현]
...

[대체 표현]
...
"""
)


# ============================================================
# 9. 메인 생성 함수
# ============================================================
def generate_recommended_reply(question: str, method: str = "rrf", k: int = 3) -> dict:
    openai_api_key = load_api_key()
    rag_df, response_df = load_dataframes()
    bm25 = build_bm25(rag_df)
    vector_db = load_vector_db(openai_api_key)
    llm = load_llm(openai_api_key)

    retrieved_docs = retrieve_documents(
        question=question,
        rag_df=rag_df,
        bm25=bm25,
        vector_db=vector_db,
        method=method,
        k=k,
    )

    situation_summary = summarize_current_situation(question, retrieved_docs)
    main_emotion = get_main_emotion(question, retrieved_docs)
    risk_level = get_main_risk_level(retrieved_docs)
    context = format_docs(retrieved_docs)

    response_examples = get_response_examples(
        response_df=response_df,
        retrieved_docs=retrieved_docs,
        emotion=main_emotion,
        question=question,
        top_n=3,
    )

    final_prompt = PROMPT.format(
        question=question,
        situation_summary=situation_summary,
        main_emotion=main_emotion,
        risk_level=risk_level,
        context=context,
        response_examples=response_examples,
    )

    result = llm.invoke(final_prompt)

    return {
        "question": question,
        "method": method,
        "retrieved_docs": retrieved_docs,
        "situation_summary": situation_summary,
        "main_emotion": main_emotion,
        "risk_level": risk_level,
        "response_examples": response_examples,
        "result_text": result.content,
    }


# ============================================================
# 10. 단독 실행 테스트
# ============================================================
def main() -> None:
    test_question = "남자친구가 내 말을 제대로 안 들어주는 것 같아서 서운해. 어떻게 보내면 좋을까?"
    output = generate_recommended_reply(test_question, method="rrf", k=3)

    print("\n===== 입력 질문 =====")
    print(output["question"])

    print("\n===== 상황 요약 =====")
    print(output["situation_summary"])

    print("\n===== 대표 감정 =====")
    print(output["main_emotion"])

    print("\n===== 갈등 위험도 =====")
    print(output["risk_level"])

    print("\n===== 최종 생성 결과 =====")
    print(output["result_text"])


if __name__ == "__main__":
    main()