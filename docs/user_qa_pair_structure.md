# 사용자 관점 질문-답변 페어 구조

## 1. 목적

이 문서는 기존 `response_pairs.csv`를 사용자 입장에서 바로 이해하고 활용할 수 있는 질문-답변 페어 구조로 재정리하기 위한 기준 문서이다.

기존 데이터는 원천 데이터의 역할명인 `speaker`와 `listener`를 사용한다. 실제 챗봇 서비스 관점에서는 `speaker`가 사용자의 말이고, `listener`가 챗봇 또는 상담형 응답자의 답변에 해당한다.

따라서 본 구조에서는 다음처럼 역할을 바꿔 해석한다.

| 원본 역할 | 사용자 관점 역할 | 의미 |
|---|---|---|
| speaker | user | 사용자가 말한 고민, 감정, 질문 |
| listener | assistant | 사용자에게 제공할 공감형 답변 |

---

## 2. 생성 파일

새로 생성하는 파생 파일:

- `data/processed/user_qa_pairs.csv`

주의:

- 원본 파일인 `data/processed/response_pairs.csv`는 수정하지 않는다.
- 원본 파일인 `data/processed/rag_documents.csv`는 수정하지 않는다.
- 기존 문서는 수정하지 않고, 본 문서처럼 별도 하위 문서로 구조 기준을 관리한다.

---

## 3. 구조 기준

`user_qa_pairs.csv`는 listener 응답 1개를 기준으로 1행을 만든다.

즉, 한 행은 다음 의미를 가진다.

> 사용자가 이런 상황과 맥락에서 이렇게 말했다.  
> 이때 assistant는 이렇게 답변한다.

---

## 4. 컬럼 정의

| 컬럼명 | 의미 | 생성 기준 |
|---|---|---|
| qa_pair_id | 질문-답변 페어 고유 ID | `dialogue_id` + 응답 순번 |
| dialogue_id | 원본 대화 ID | `response_pairs.csv`의 `dialogue_id` |
| response_index | 한 대화 안에서 몇 번째 assistant 응답인지 | `dialogue_id`별 누적 순번 |
| relation | 관계 유형 | 원본 `relation` |
| situation | 사용자가 처한 상황 요약 | 원본 `situation` |
| user_emotion | 사용자 감정 | 원본 `speaker_emotion` |
| risk_level | 갈등 위험도 | `rag_documents.csv`에서 `dialogue_id` 기준으로 연결 |
| user_context | 답변 직전까지의 대화 맥락 | `speaker:`는 `user:`, `listener:`는 `assistant:`로 변환 |
| user_question | assistant 답변 직전의 마지막 사용자 발화 | `context_before_response`에서 마지막 `speaker:` 발화 추출 |
| assistant_answer | 추천 답변 예시 | 원본 `listener_response` |
| answer_empathy | 답변 공감 유형 | 원본 `listener_empathy` |
| is_terminal | 해당 답변 후 대화 종료 여부 | 원본 `terminate` |
| grade | 원본 대화 품질 등급 | `rag_documents.csv`에서 연결 |
| avg_rating | 원본 대화 평균 평점 | `rag_documents.csv`에서 연결 |
| final_speaker_change_emotion | 대화 종료 시 사용자 감정 변화 | `rag_documents.csv`에서 연결 |

---

## 5. 예시 형태

```text
user_context:
user: 자기야, 나 결국 돈 아끼기에 실패했어.
assistant: 어라, 왜 그래? 큰돈 나갈 일이 있었어?
user: 아니, 요즘 미세먼지가 너무 심해서 결국 공기청정기를 샀잖아.

user_question:
아니, 요즘 미세먼지가 너무 심해서 결국 공기청정기를 샀잖아.

assistant_answer:
한참 고민하더니 결국 샀구나? 공기는 안 좋고, 공기청정기 금액은 싸지도 않고, 고민 많았을 텐데.
```

---

## 6. 활용 기준

### RAG 검색

- `situation`
- `user_emotion`
- `risk_level`
- `user_context`
- `user_question`

위 컬럼을 검색 쿼리 또는 메타데이터 필터로 활용한다.

### 답변 생성 참고

- `assistant_answer`
- `answer_empathy`
- `grade`
- `avg_rating`

위 컬럼을 실제 답변 예시와 품질 기준으로 활용한다.

### 사용자 화면/챗봇 관점

서비스 화면에서는 `speaker/listener`라는 데이터셋 내부 용어를 노출하지 않는다.

대신 다음 표현을 사용한다.

- 사용자 입력: `user_question`
- 이전 대화 맥락: `user_context`
- 추천 답변: `assistant_answer`
- 공감 유형: `answer_empathy`

---

## 7. 생성 방식

생성 스크립트:

- `src/build_user_qa_pairs.py`
- `src/build_user_qa_pairs.ps1`

실행 결과:

- `data/processed/user_qa_pairs.csv`

처리 원칙:

1. 원본 CSV는 읽기만 한다.
2. 기존 파일을 덮어쓰지 않는다.
3. 사용자 관점의 새 파생 파일만 생성한다.
4. 역할명은 `speaker/listener`에서 `user/assistant`로 바꿔 저장한다.
