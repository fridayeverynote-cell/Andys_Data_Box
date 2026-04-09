from pathlib import Path
import json
import pandas as pd


# =========================================================
# 0. 경로 설정
# - 프로젝트 루트 / data / outputs 경로를 잡는다.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "training"
VALID_DIR = DATA_DIR / "validation"
OUTPUT_DIR = BASE_DIR / "outputs"

# outputs 폴더가 없으면 자동 생성
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 1. JSON 파일 로드 함수
# - training, validation 폴더 안의 json 파일들을 읽음.
# - 깨진 파일이 있어도 전체 코드가 멈추지 않도록 try-except 처리.
# =========================================================
def load_json_files(folder_path: Path):
    json_files = list(folder_path.glob("*.json"))
    results = []

    print(f"[LOAD] {folder_path} -> {len(json_files)} files")

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")

    return results


# =========================================================
# 2. 대화 텍스트 정리 함수들
# - utterances를 바탕으로 전체 대화, speaker 발화, listener 발화를 각각 만듦.
# - 이후 RAG 검색 / 응답 추천 / 분석용으로 사용됨.
# =========================================================
def join_full_dialogue(utterances):
    """
    대화 전체를 'role: text' 형식으로 합침.
    예:
    speaker: 오늘 힘들었어
    listener: 무슨 일 있었어?
    """
    lines = []

    for u in utterances:
        role = (u.get("role", "") or "").strip()
        text = (u.get("text", "") or "").strip()

        if role and text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines)


def extract_speaker_texts(utterances):
    """
    speaker가 한 말만 순서대로 이어 붙임.
    사용자 고민/감정 분석에 활용하기 좋음.
    """
    texts = []

    for u in utterances:
        if u.get("role") == "speaker":
            text = (u.get("text", "") or "").strip()
            if text:
                texts.append(text)

    return " ".join(texts)


def extract_listener_texts(utterances):
    """
    listener가 한 말만 순서대로 이어 붙임.
    공감 응답 패턴 분석에 활용하기 좋음.
    """
    texts = []

    for u in utterances:
        if u.get("role") == "listener":
            text = (u.get("text", "") or "").strip()
            if text:
                texts.append(text)

    return " ".join(texts)


# =========================================================
# 3. 공감 태그 / 감정 변화 추출 함수
# - listener_empathy: listener 발화에 붙은 공감 태그들을 모음.
# - speaker_changeEmotion: 대화 끝나며 화자 감정이 변했는지 확인.
# =========================================================
def extract_listener_empathy_tags(utterances):
    """
    listener 발화들에 달린 listener_empathy 태그를 전부 모아 중복 제거.
    값이 없으면 빈 문자열이 되도록 나중에 처리.
    """
    tags = []

    for u in utterances:
        if u.get("role") == "listener":
            empathy_list = u.get("listener_empathy")
            if empathy_list:
                tags.extend(empathy_list)

    return sorted(set(tags))


def get_final_speaker_change_emotion(utterances):
    """
    뒤에서부터 보면서 마지막으로 등장한 speaker_changeEmotion 값을 가져옴.
    없으면 빈 문자열("") 반환.
    """
    for u in reversed(utterances):
        change_emotion = u.get("speaker_changeEmotion")
        if change_emotion:
            return change_emotion

    return ""


# =========================================================
# 4. 위험도(risk_level) 판별 함수
# - 현재는 규칙 기반으로 간단히 high / normal만 구분.
# - 일반 연인 갈등과 건강/죽음/폭력 등 고위험 상황을 구분하기 위한 시작점.
# =========================================================
def detect_risk_level(info, utterances):
    """
    situation + 전체 발화 텍스트를 합쳐서 위험 키워드가 있으면 high로 분류.
    """
    situation = info.get("situation", "") or ""
    all_text = " ".join((u.get("text", "") or "") for u in utterances)

    combined_text = f"{situation} {all_text}"

    # 고위험으로 우선 분류할 키워드
    high_risk_keywords = [
        "시한부", "영정사진", "암", "죽고", "죽을", "죽음", "죽는다",
        "자살", "극단적", "폭력", "학대", "협박", "성폭행", "폭행"
    ]

    for keyword in high_risk_keywords:
        if keyword in combined_text:
            return "high"

    return "normal"


# =========================================================
# 5. RAG용 데이터프레임 생성
# - 대화 1개당 1행으로 만듦.
# - 이후 벡터DB에 넣기 좋은 형태의 기본 테이블
# =========================================================
def build_rag_dataframe(json_list):
    rows = []

    for data in json_list:
        info = data.get("info", {})
        utterances = data.get("utterances", [])

        # 연인 관계만 사용
        if info.get("relation") != "연인":
            continue

        rows.append({
            "dialogue_id": info.get("id"),
            "file_name": info.get("name"),
            "relation": info.get("relation"),
            "situation": info.get("situation", ""),
            "speaker_emotion": info.get("speaker_emotion", ""),
            "listener_behavior": ", ".join(info.get("listener_behavior", [])),
            "avg_rating": info.get("evaluation", {}).get("avg_rating"),
            "grade": info.get("evaluation", {}).get("grade"),

            # 대화 본문 관련 컬럼
            "speaker_texts": extract_speaker_texts(utterances),
            "listener_texts": extract_listener_texts(utterances),
            "full_dialogue": join_full_dialogue(utterances),

            # 태그 / 상태 관련 컬럼
            "listener_empathy_tags": ", ".join(extract_listener_empathy_tags(utterances)),
            "final_speaker_change_emotion": get_final_speaker_change_emotion(utterances),
            "risk_level": detect_risk_level(info, utterances),

            # 메타 정보
            "turn_count": len(utterances),
            "terminated": utterances[-1].get("terminate") if utterances else False,
        })

    rag_df = pd.DataFrame(rows)

    # -----------------------------------------------------
    # NaN / 빈값 정리
    # - 나중에 Document metadata, JSON 변환, 프롬프트 구성 시 편하도록 정리
    # -----------------------------------------------------
    text_columns = [
        "file_name", "situation", "speaker_emotion", "listener_behavior",
        "speaker_texts", "listener_texts", "full_dialogue",
        "listener_empathy_tags", "final_speaker_change_emotion", "risk_level"
    ]

    for col in text_columns:
        if col in rag_df.columns:
            rag_df[col] = rag_df[col].fillna("").astype(str)

    # -----------------------------------------------------
    # 최종 결측 기준 반영
    # - 빈 문자열이면 unknown으로 통일함
    # -----------------------------------------------------
    if "final_speaker_change_emotion" in rag_df.columns:
        rag_df["final_speaker_change_emotion"] = (
            rag_df["final_speaker_change_emotion"].replace("", "unknown")
        )

    return rag_df


# =========================================================
# 6. 응답 pair 데이터프레임 생성
# - listener 응답 1개당 1행으로 만듦.
# - 어떤 맥락에서 어떤 공감 응답이 나왔는지 학습/추천하기 용이함.
# =========================================================
def build_response_pair_dataframe(json_list):
    rows = []

    for data in json_list:
        info = data.get("info", {})
        utterances = data.get("utterances", [])

        # 연인 관계만 사용
        if info.get("relation") != "연인":
            continue

        context_buffer = []

        for u in utterances:
            role = u.get("role")
            text = (u.get("text", "") or "").strip()

            # listener 응답이 나올 때마다, 그 직전까지의 맥락을 함께 저장
            if role == "listener" and text:
                rows.append({
                    "dialogue_id": info.get("id"),
                    "relation": info.get("relation", ""),
                    "situation": info.get("situation", ""),
                    "speaker_emotion": info.get("speaker_emotion", ""),
                    "context_before_response": "\n".join(context_buffer),
                    "listener_response": text,
                    "listener_empathy": ", ".join(u.get("listener_empathy", [])) if u.get("listener_empathy") else "",
                    "terminate": u.get("terminate", False),
                })

            # 대화 흐름 유지용 버퍼
            if role and text:
                context_buffer.append(f"{role}: {text}")

    pair_df = pd.DataFrame(rows)

    # -----------------------------------------------------
    # NaN / 빈값 정리
    # -----------------------------------------------------
    text_columns = [
        "relation", "situation", "speaker_emotion",
        "context_before_response", "listener_response", "listener_empathy"
    ]

    for col in text_columns:
        if col in pair_df.columns:
            pair_df[col] = pair_df[col].fillna("").astype(str)

    # -----------------------------------------------------
    # 최종 결측 기준 반영
    # - 빈 문자열이면 미분류로 통일함
    # -----------------------------------------------------
    if "listener_empathy" in pair_df.columns:
        pair_df["listener_empathy"] = (
            pair_df["listener_empathy"].replace("", "미분류")
        )

    return pair_df


# =========================================================
# 7. 데이터 품질 요약 출력 함수
# - CSV 저장 전에 간단한 상태를 확인
# =========================================================
def print_basic_summary(rag_df, pair_df):
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)

    print(f"연인 대화 수: {len(rag_df)}")
    print(f"응답 pair 수: {len(pair_df)}")

    print("\n[rag_documents.csv - risk_level 분포]")
    print(rag_df["risk_level"].value_counts(dropna=False))

    print("\n[rag_documents.csv - final_speaker_change_emotion 'unknown' 개수]")
    print((rag_df["final_speaker_change_emotion"] == "unknown").sum())

    print("\n[response_pairs.csv - listener_empathy '미분류' 개수]")
    print((pair_df["listener_empathy"] == "미분류").sum())

    print("\n[rag_documents.csv - shape]")
    print(rag_df.shape)

    print("\n[response_pairs.csv - shape]")
    print(pair_df.shape)


# =========================================================
# 8. 메인 실행 함수
# - training + validation 로드
# - 두 데이터를 합친 뒤 전처리
# - csv 저장
# =========================================================
def main():
    # 1) training / validation 데이터 로드
    train_data = load_json_files(TRAIN_DIR)
    valid_data = load_json_files(VALID_DIR)

    # 2) 두 폴더 데이터를 합쳐서 전체 데이터 구성
    all_data = train_data + valid_data
    print(f"\n[INFO] total loaded json: {len(all_data)}")

    # 3) 두 종류의 결과 테이블 생성
    rag_df = build_rag_dataframe(all_data)
    pair_df = build_response_pair_dataframe(all_data)

    # 4) 저장 경로 설정
    rag_output_path = OUTPUT_DIR / "rag_documents.csv"
    pair_output_path = OUTPUT_DIR / "response_pairs.csv"

    # 5) csv 저장
    rag_df.to_csv(rag_output_path, index=False, encoding="utf-8-sig")
    pair_df.to_csv(pair_output_path, index=False, encoding="utf-8-sig")

    # 6) 저장 결과 출력
    print(f"\n[SAVE] {rag_output_path}")
    print(f"[SAVE] {pair_output_path}")

    # 7) 샘플 확인
    print("\n[RAG DF SAMPLE]")
    print(rag_df.head(3))

    print("\n[PAIR DF SAMPLE]")
    print(pair_df.head(3))

    # 8) 요약 통계 출력
    print_basic_summary(rag_df, pair_df)


# =========================================================
# 9. 실행 진입점
# =========================================================
if __name__ == "__main__":
    main()