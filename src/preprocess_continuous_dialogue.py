from pathlib import Path
import pandas as pd


# =========================================================
# 0. 경로 설정
# - 현재 파일 기준으로 프로젝트 루트 찾기
# - data 폴더와 outputs 폴더를 경로로 잡음
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# outputs 폴더가 없으면 생성
OUTPUT_DIR.mkdir(exist_ok=True)

# 원본 엑셀 파일 경로
RAW_FILE_PATH = DATA_DIR / "한국어_연속적_대화_데이터셋.xlsx"

# 최종 저장 경로
UTTERANCE_OUTPUT_PATH = OUTPUT_DIR / "continuous_dialogue_utterance.csv"
DIALOGUE_OUTPUT_PATH = OUTPUT_DIR / "continuous_dialogue_dialogue.csv"


# =========================================================
# 1. 엑셀 파일 읽기
# - header 없이 전체 원본 그대로 읽음
# - 상단 메타 정보까지 포함해서 읽음
# =========================================================
def load_excel_file(file_path: Path) -> pd.DataFrame:
    df_raw = pd.read_excel(file_path, header=None)
    return df_raw


# =========================================================
# 2. 원본 데이터 기본 정리
# - 실제 데이터 구간만 남김
# - 필요한 3개 컬럼만 사용
# - 컬럼명 새로 지정
# =========================================================
def clean_raw_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    # 실제 데이터는 3번째 줄부터 시작
    df = df_raw.iloc[2:, [0, 1, 2]].copy()

    # 컬럼명 새로 지정
    df.columns = ["dialog_marker", "utterance", "emotion"]

    # 발화가 비어 있는 행 제거
    df = df.dropna(subset=["utterance"]).copy()

    # 문자열 양쪽 공백 제거
    df["dialog_marker"] = df["dialog_marker"].fillna("").astype(str).str.strip()
    df["utterance"] = df["utterance"].fillna("").astype(str).str.strip()
    df["emotion"] = df["emotion"].fillna("").astype(str).str.strip()

    # 빈 문자열 발화 제거
    df = df[df["utterance"] != ""].copy()

    return df


# =========================================================
# 3. 대화 그룹 번호 생성
# - dialog_marker가 S이면 새 대화 시작 의미
# - 누적합으로 dialogue_group_id 생성
# - 같은 대화 안에서 turn 번호 생성
# =========================================================
def add_dialogue_group_info(df: pd.DataFrame) -> pd.DataFrame:
    # 새 대화 시작 여부 표시
    df["is_new_dialogue"] = df["dialog_marker"].eq("S").astype(int)

    # 누적합으로 대화 그룹 번호 생성
    df["dialogue_group_id"] = df["is_new_dialogue"].cumsum()

    # turn 순서 생성
    df["turn_index"] = df.groupby("dialogue_group_id").cumcount() + 1

    return df


# =========================================================
# 4. 발화 단위 데이터프레임 생성
# - 감정 분석용 기본 작업본
# - 한 행이 발화 1개 구조
# =========================================================
def build_utterance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    utterance_df = df[["dialogue_group_id", "turn_index", "utterance", "emotion"]].copy()

    utterance_df = utterance_df.sort_values(
        by=["dialogue_group_id", "turn_index"]
    ).reset_index(drop=True)

    return utterance_df


# =========================================================
# 5. emotion 값 기본 정리
# - 공백 제거
# - 문자열 통일
# =========================================================
def normalize_emotion_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["emotion"] = df["emotion"].fillna("").astype(str).str.strip()
    return df


# =========================================================
# 6. emotion 오타 수정
# - 수정 가능한 값만 치환
# =========================================================
def fix_emotion_typos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    typo_map = {
        "ㅈ중립": "중립",
        "ㄴ중립": "중립",
        "중림": "중립",
        "분ㄴ": "분노",
        "분": "분노",
    }

    df["emotion"] = df["emotion"].replace(typo_map)

    return df


# =========================================================
# 7. 정상 감정만 남김
# - 애매한 값 제거
# =========================================================
def keep_valid_emotions_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_emotions = {"중립", "놀람", "분노", "슬픔", "행복", "혐오", "공포"}

    # 정상 감정만 유지
    df = df[df["emotion"].isin(valid_emotions)].copy()

    # index 재정렬
    df = df.reset_index(drop=True)

    return df


# =========================================================
# 8. turn_index 재생성
# - 일부 발화 제거 후 turn 번호가 끊길 수 있어서 다시 생성
# =========================================================
def rebuild_turn_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values(["dialogue_group_id", "turn_index"]).reset_index(drop=True)
    df["turn_index"] = df.groupby("dialogue_group_id").cumcount() + 1

    return df


# =========================================================
# 9. 대화 단위 데이터프레임 생성
# - 같은 대화끼리 묶어서 한 행으로 만듦
# - 대화 흐름 분석용 보조 데이터
# =========================================================
def build_dialogue_dataframe(utterance_df: pd.DataFrame) -> pd.DataFrame:
    dialogue_df = (
        utterance_df.groupby("dialogue_group_id")
        .agg(
            full_dialogue=("utterance", lambda x: " [SEP] ".join(x)),
            emotion_sequence=("emotion", lambda x: " > ".join(x)),
            turn_count=("turn_index", "max")
        )
        .reset_index()
    )

    return dialogue_df


# =========================================================
# 10. 저장 함수
# - utf-8-sig로 저장됨
# - 엑셀에서 한글 깨짐 줄이기 위함
# =========================================================
def save_dataframe(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {output_path}")


# =========================================================
# 11. 최종 요약 출력
# - 정리 전후 비교
# - 감정 분포 / 대화 길이 / 샘플 확인
# =========================================================
def print_summary(before_df: pd.DataFrame, utterance_df: pd.DataFrame, dialogue_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)

    print(f"정리 전 발화 수: {len(before_df)}")
    print(f"정리 후 발화 수: {len(utterance_df)}")
    print(f"제거된 발화 수: {len(before_df) - len(utterance_df)}")
    print(f"정리 후 대화 수: {len(dialogue_df)}")

    print("\n[정리 후 emotion 분포]")
    print(utterance_df["emotion"].value_counts())

    print("\n[대화 길이 요약]")
    print(dialogue_df["turn_count"].describe())

    print("\n[정리 후 발화 샘플]")
    print(utterance_df.head(5))

    print("\n[정리 후 대화 샘플]")
    print(dialogue_df.head(3))

    print("\n[결측치 확인]")
    print(utterance_df.isna().sum())
    print(dialogue_df.isna().sum())


# =========================================================
# 12. 메인 실행 함수
# - 원본 엑셀 로드
# - 기본 전처리
# - 감정 라벨 정리
# - 최종 CSV 저장
# =========================================================
def main():
    # 1) 원본 엑셀 읽기
    df_raw = load_excel_file(RAW_FILE_PATH)

    # 2) 원본 데이터 기본 정리
    df_clean = clean_raw_dataframe(df_raw)

    # 정리 전 발화 수 비교용 백업
    before_df = df_clean.copy()

    # 3) 대화 그룹 번호와 turn 번호 생성
    df_grouped = add_dialogue_group_info(df_clean)

    # 4) 발화 단위 데이터 생성
    utterance_df = build_utterance_dataframe(df_grouped)

    # 5) emotion 문자열 정리
    utterance_df = normalize_emotion_text(utterance_df)

    # 6) emotion 오타 수정
    utterance_df = fix_emotion_typos(utterance_df)

    # 7) 정상 감정만 유지
    utterance_df = keep_valid_emotions_only(utterance_df)

    # 8) turn_index 재생성
    utterance_df = rebuild_turn_index(utterance_df)

    # 9) 대화 단위 데이터 생성
    dialogue_df = build_dialogue_dataframe(utterance_df)

    # 10) 최종 CSV 저장
    save_dataframe(utterance_df, UTTERANCE_OUTPUT_PATH)
    save_dataframe(dialogue_df, DIALOGUE_OUTPUT_PATH)

    # 11) 요약 출력
    print_summary(before_df, utterance_df, dialogue_df)


# =========================================================
# 13. 실행 진입점
# =========================================================
if __name__ == "__main__":
    main()