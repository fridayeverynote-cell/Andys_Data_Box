# ============================================================
# - rag_documents.csv / response_pairs.csv 컬럼 구조 확인
# - 검색 품질 개선을 위한 rag_text 재구성
# - 답변 예시 텍스트 생성
# - 중간 결과 CSV 저장
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 1. 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAG_PATH = RAW_DATA_DIR / "rag_documents.csv"
RESPONSE_PATH = RAW_DATA_DIR / "response_pairs.csv"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CSV 로드
# ============================================================
rag_df = pd.read_csv(RAG_PATH)
response_df = pd.read_csv(RESPONSE_PATH)

print("===== rag_documents 기본 정보 =====")
print(rag_df.shape)
print(rag_df.columns.tolist())

print("\n===== response_pairs 기본 정보 =====")
print(response_df.shape)
print(response_df.columns.tolist())


# ============================================================
# 3. 문자열 처리 함수
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def short_text(text, max_len=280):
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len]


# ============================================================
# 4. rag_text 생성 함수
# - 검색 정확도를 위해 상황 중심으로 재구성함
# - 감정/청자반응/핵심 발화 일부만 사용함
# - full_dialogue는 사용하지 않음
# ============================================================
def build_rag_text(row):
    parts = []

    relation = clean_text(row["relation"])
    situation = clean_text(row["situation"])
    speaker_emotion = clean_text(row["speaker_emotion"])
    listener_behavior = clean_text(row["listener_behavior"])
    empathy_tags = clean_text(row["listener_empathy_tags"])
    risk_level = clean_text(row["risk_level"])

    speaker_texts = short_text(row["speaker_texts"], max_len=280)
    listener_texts = short_text(row["listener_texts"], max_len=280)

    if relation:
        parts.append(f"관계: {relation}")

    if situation:
        parts.append(f"핵심 상황: {situation}")

    if speaker_emotion:
        parts.append(f"주요 감정: {speaker_emotion}")

    if listener_behavior:
        parts.append(f"청자 반응 특성: {listener_behavior}")

    if empathy_tags:
        parts.append(f"공감 유형: {empathy_tags}")

    if risk_level:
        parts.append(f"위험도: {risk_level}")

    if speaker_texts:
        parts.append(f"화자 핵심 발화 예시: {speaker_texts}")

    if listener_texts:
        parts.append(f"청자 핵심 발화 예시: {listener_texts}")

    return "\n".join(parts)


# ============================================================
# 5. response example 생성 함수
# - 4단계 프롬프트용 참고 예시를 만듦
# - 문맥은 너무 길지 않게 줄임
# ============================================================
def build_response_example_text(row):
    parts = []

    relation = clean_text(row["relation"])
    situation = clean_text(row["situation"])
    speaker_emotion = clean_text(row["speaker_emotion"])
    context_before_response = short_text(row["context_before_response"], max_len=350)
    listener_response = clean_text(row["listener_response"])
    listener_empathy = clean_text(row["listener_empathy"])
    terminate = clean_text(row["terminate"])

    if relation:
        parts.append(f"관계: {relation}")

    if situation:
        parts.append(f"상황: {situation}")

    if speaker_emotion:
        parts.append(f"화자 감정: {speaker_emotion}")

    if context_before_response:
        parts.append(f"응답 직전 문맥: {context_before_response}")

    if listener_response:
        parts.append(f"추천 가능한 청자 응답 예시: {listener_response}")

    if listener_empathy:
        parts.append(f"응답 공감 유형: {listener_empathy}")

    if terminate:
        parts.append(f"대화 종료 여부: {terminate}")

    return "\n".join(parts)


# ============================================================
# 6. 텍스트 컬럼 생성
# ============================================================
rag_df["rag_text"] = rag_df.apply(build_rag_text, axis=1)
response_df["response_example_text"] = response_df.apply(build_response_example_text, axis=1)


# ============================================================
# 7. 결과 확인
# ============================================================
print("\n===== 개선된 rag_text 샘플 =====")
print(rag_df.loc[0, "rag_text"])

print("\n===== response_example_text 샘플 =====")
print(response_df.loc[0, "response_example_text"])


# ============================================================
# 8. 저장
# ============================================================
rag_output_path = PROCESSED_DATA_DIR / "rag_documents_with_text.csv"
response_output_path = PROCESSED_DATA_DIR / "response_pairs_with_text.csv"

rag_df.to_csv(rag_output_path, index=False, encoding="utf-8-sig")
response_df.to_csv(response_output_path, index=False, encoding="utf-8-sig")

print("\n===== 저장 완료 =====")
print(rag_output_path)
print(response_output_path)