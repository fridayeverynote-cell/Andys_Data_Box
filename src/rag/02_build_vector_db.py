# ============================================================
# - rag_documents_with_text.csv를 불러옴
# - rag_text를 임베딩함
# - FAISS Dense 벡터DB를 생성하고 저장함
# - metadata를 풍부하게 함께 저장함
# ============================================================

from pathlib import Path
import os
import time
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAG_TEXT_PATH = PROCESSED_DATA_DIR / "rag_documents_with_text.csv"
VECTOR_DB_DIR = PROCESSED_DATA_DIR / "faiss_rag_db"

MAX_TEXT_LENGTH = 4000


# ============================================================
# 2. 환경변수 로드
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않음.")


# ============================================================
# 3. CSV 로드
# ============================================================
if not RAG_TEXT_PATH.exists():
    raise FileNotFoundError(f"파일을 찾을 수 없음: {RAG_TEXT_PATH}")

rag_df = pd.read_csv(RAG_TEXT_PATH)

print("===== CSV 로드 완료 =====")
print(rag_df.shape)
print(rag_df.columns.tolist())


# ============================================================
# 4. 문자열 처리 함수
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def truncate_text(text, max_len=4000):
    text = clean_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len]


# ============================================================
# 5. rag_text 유효성 확인
# ============================================================
if "rag_text" not in rag_df.columns:
    raise ValueError("rag_text 컬럼이 없음.")

rag_df["rag_text"] = rag_df["rag_text"].astype(str).str.strip()
rag_df = rag_df[rag_df["rag_text"] != ""].copy()

print("\n===== 유효 문서 수 =====")
print(len(rag_df))


# ============================================================
# 6. texts / metadatas 준비
# - Dense retrieval에서 검색 결과 설명용 metadata도 같이 넣음
# ============================================================
texts = []
metadatas = []

for _, row in rag_df.iterrows():
    text = truncate_text(row["rag_text"], MAX_TEXT_LENGTH)
    texts.append(text)

    metadatas.append({
        "dialogue_id": clean_text(row.get("dialogue_id", "")),
        "file_name": clean_text(row.get("file_name", "")),
        "relation": clean_text(row.get("relation", "")),
        "situation": clean_text(row.get("situation", "")),
        "speaker_emotion": clean_text(row.get("speaker_emotion", "")),
        "listener_behavior": clean_text(row.get("listener_behavior", "")),
        "listener_empathy_tags": clean_text(row.get("listener_empathy_tags", "")),
        "risk_level": clean_text(row.get("risk_level", "")),
        "conflict_keywords": clean_text(row.get("conflict_keywords", "")),
        "turn_count": clean_text(row.get("turn_count", "")),
        "terminated": clean_text(row.get("terminated", "")),
    })

print("\n===== 첫 번째 metadata 샘플 =====")
print(metadatas[0])


# ============================================================
# 7. 임베딩 모델 준비
# ============================================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)


# ============================================================
# 8. FAISS 벡터DB 생성
# ============================================================
print("\n===== FAISS 벡터DB 생성 시작 =====")
start_time = time.time()

vector_db = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
)

elapsed = time.time() - start_time

print("===== FAISS 벡터DB 생성 완료 =====")
print(f"소요 시간: {round(elapsed, 2)}초")


# ============================================================
# 9. 저장
# ============================================================
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
vector_db.save_local(str(VECTOR_DB_DIR))

print("\n===== 저장 완료 =====")
print(VECTOR_DB_DIR)