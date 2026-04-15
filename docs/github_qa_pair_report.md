# 🚀 [Feature] 사용자 관점 질문-답변(QA) 페어 데이터 파이프라인 구축

## 📝 개요 (Overview)
기존 정제 데이터(`response_pairs.csv` 및 `rag_documents.csv`)를 결합하여, 실제 챗봇 서비스 및 RAG(검색 증강 생성) 시스템에서 즉시 활용할 수 있는 **사용자 중심의 QA 페어 데이터(`user_qa_pairs.csv`) 파이프라인**을 구축했습니다.

이전 데이터셋의 원천 용어(`speaker`, `listener`)를 서비스 관점에 맞게(`user`, `assistant`) 치환하고, 단일 응답 단위로 데이터를 평탄화하여 활용성을 극대화했습니다.

---

## ✨ 주요 변경 및 구축 사항 (Key Updates)

### 1. 역할명(Role) 및 컨텍스트 치환
* 서비스 UI/UX와 직관적으로 매칭되도록 대화 주체의 명칭을 변경했습니다.
  * `speaker` ➡️ **`user`** (사용자의 고민, 감정, 질문)
  * `listener` ➡️ **`assistant`** (제공할 공감형 답변)

### 2. 단일 응답(Assistant Answer) 단위의 행 구성
* **1 Assistant Answer = 1 Row** 원칙을 적용했습니다.
* **의미:** *"사용자가 특정 맥락(`user_context`)과 상황(`situation`)에서 마지막으로 이렇게 말했을 때(`user_question`), 시스템은 이렇게 답변한다(`assistant_answer`)."*

### 3. RAG 메타데이터 결합 (Data Enrichment)
* 기존 `response_pairs`에 `rag_documents`를 Join하여 종합적인 메타데이터를 포함했습니다.
  * 갈등 위험도 (`risk_level`)
  * 대화 품질 평가 (`grade`, `avg_rating`)
  * 대화 종료 후 사용자 감정 변화 (`final_speaker_change_emotion`)

---

## 📊 데이터셋 스키마 요약 (Schema Summary)

| 분류 | 컬럼명 | 설명 |
|:---:|:---|:---|
| **기본 정보** | `qa_pair_id` | 질문-답변 페어 고유 ID (`dialogue_id` + 응답 순번) |
| | `dialogue_id` | 원본 대화 ID |
| **상태/속성** | `relation` / `situation` | 관계 유형 및 사용자가 처한 상황 요약 |
| | `user_emotion` | 사용자의 현재 감정 |
| | `risk_level` | 갈등 위험도 파악 (normal, high 등) |
| **대화 텍스트** | `user_context` | 답변 직전까지의 대화 맥락 전체 (`user`, `assistant` 치환 완료) |
| | `user_question` | assistant 답변 직전의 **마지막 사용자 발화** (RAG 쿼리용) |
| | `assistant_answer` | 추천 공감형 답변 예시 |
| **품질/평가** | `answer_empathy` | 답변의 공감 유형 |
| | `grade` / `avg_rating` | 해당 대화의 품질 등급 및 평점 |

---

## 💡 서비스 활용 방안 (Usage Guide)

이번에 생성된 `user_qa_pairs.csv`는 다음과 같은 영역에 직접 투입됩니다.

### 🔍 1. RAG (검색 증강 생성)
* **검색 쿼리**: `user_question`을 벡터화하여 질문 의도 검색
* **필터링 메타데이터**: `situation`, `user_emotion`, `risk_level`을 필터 조건으로 활용하여 맥락에 맞는 답변 반환율 향상

### 💬 2. 프롬프트 및 응답 생성
* **Few-shot 프롬프트**: `assistant_answer` 및 `answer_empathy`를 LLM의 Few-shot 예시로 제공하여 응답 품질(Grade) 보장
* **평가 지표**: `grade`와 `avg_rating`이 높은 답변을 우선적으로 참조

### 💻 3. Streamlit 프론트엔드 (UI)
* 서비스 UI 렌더링 시 내부 용어인 `speaker/listener`를 완전히 배제하고, `user_context`를 파싱하여 깔끔한 말풍선 UI 즉시 구현 가능

---

## 🛠 실행 방법 (How to Build)

데이터 빌드 스크립트는 프로젝트 `src/utils/`에 위치하며, 기존 원본 데이터는 수정하지 않고 파생 데이터베이스만 읽기 전용으로 안전하게 생성합니다.

```bash
# 옵션: 기본 설정값으로 전체 QA 페어 데이터 재구축
python src/utils/build_user_qa_pairs.py
```

* **출력 파일**: `data/processed/user_qa_pairs.csv`
