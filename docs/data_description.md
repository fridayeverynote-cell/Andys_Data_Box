# 데이터 설명 문서

## 1. 문서 목적
이 문서는 프로젝트에서 사용하는 전처리 데이터 4개의 역할, 구조, 활용 방향을 설명하기 위한 문서이다.

프로젝트는 연인 갈등 상황 대응을 위한 감정 기반 RAG 챗봇이며, 데이터는 감정 분석용과 RAG 검색/답변 추천용으로 역할이 나뉜다.

---

## 2. 전체 데이터 역할 요약

### 감정 분석/위험도 분석용
- `continuous_dialogue_dialogue.csv`
- `continuous_dialogue_utterance.csv`

### RAG 검색/답변 추천용
- `rag_documents.csv`
- `response_pairs.csv`

---

## 3. 파일별 설명

## 3-1. continuous_dialogue_dialogue.csv

### 목적
대화 전체 단위의 감정 흐름을 파악하기 위한 파일이다.

### 생성 방식
통합 전처리 스크립트에서 발화 단위 최종본을 기준으로 다시 생성되는 최종 대화 단위 파일이다.

### 주요 활용
- 대화 분위기 분석
- 감정 흐름 파악
- 위험도 판단 보조

### 컬럼
- `dialogue_group_id`: 대화 그룹 ID
- `full_dialogue`: 대화 전체 텍스트
- `emotion_sequence`: 대화에서 나타난 감정 흐름
- `turn_count`: 발화 수

### 주 사용 담당
- 감정 분석 담당

---

## 3-2. continuous_dialogue_utterance.csv

### 목적
발화 단위 감정 분석을 위한 파일이다.

### 생성 방식
통합 전처리 스크립트에서 원본 엑셀 데이터를 정리한 뒤, 감정 문자열 정리와 오타 수정, 정상 감정 필터링까지 거쳐 생성되는 최종 발화 단위 파일이다.

### 주요 활용
- 감정 분류
- 위험 표현 탐색
- 사용자 입력 감정 예측 보조

### 컬럼
- `dialogue_group_id`: 대화 그룹 ID
- `turn_index`: 발화 순서
- `utterance`: 발화 텍스트
- `emotion`: 감정 라벨

### 주 사용 담당
- 감정 분석 담당

---

## 3-3. rag_documents.csv

### 목적
유사 사례 검색을 위한 RAG 핵심 문서 파일이다.

### 생성 방식
공감형 대화 데이터에서 연인 관계 대화만 추출하고, 대화 1개당 1행 구조로 정리한 최종 검색용 문서 파일이다.

### 주요 활용
- 유사 갈등 상황 검색
- 추천 답변 생성의 근거 문서 제공
- 위험도 참조

### 컬럼
- `dialogue_id`: 대화 ID
- `file_name`: 원본 파일 식별 정보
- `relation`: 관계 유형
- `situation`: 상황 요약
- `speaker_emotion`: 화자 감정
- `listener_behavior`: 청자 반응 방식 요약
- `avg_rating`: 평균 평점
- `grade`: 등급
- `speaker_texts`: 화자 발화 모음
- `listener_texts`: 청자 발화 모음
- `full_dialogue`: 전체 대화
- `listener_empathy_tags`: 공감 태그
- `final_speaker_change_emotion`: 최종 감정 변화
- `risk_level`: 갈등 위험도
- `turn_count`: 발화 수
- `terminated`: 종료 여부

### 주 사용 담당
- RAG 담당
- 답변 추천 담당
- 통합 담당

### 참고
현재 `rag_documents.csv`는 결측 처리 기준까지 반영된 최종 사용 파일이다.
`final_speaker_change_emotion`의 기존 빈값은 `unknown`으로 통일되었다.

---

## 3-4. response_pairs.csv

### 목적
문맥-응답 기반 추천 답변 예시를 제공하기 위한 파일이다.

### 생성 방식
공감형 대화에서 listener 응답이 나올 때마다, 응답 직전 맥락과 실제 응답을 1행으로 정리한 최종 pair 파일이다.

### 주요 활용
- 실제 응답 예시 제공
- 문맥 기반 답변 추천
- 공감 반응 패턴 분석

### 컬럼
- `dialogue_id`: 대화 ID
- `relation`: 관계 유형
- `situation`: 상황 요약
- `speaker_emotion`: 화자 감정
- `context_before_response`: 응답 직전 문맥
- `listener_response`: 실제 청자 응답
- `listener_empathy`: 공감 유형
- `terminate`: 종료 여부

### 주 사용 담당
- 답변 추천 담당
- RAG 담당
- 통합 담당

### 참고
현재 `response_pairs.csv`는 결측 처리 기준까지 반영된 최종 사용 파일이다.
`listener_empathy`의 기존 빈값은 `미분류`로 통일되었다.

---

## 4. 협업 시 주의점

### 주의점 1
감정 분석용 파일과 RAG용 파일은 역할이 다르므로 섞어서 쓰지 않는다.

### 주의점 2
`terminated`와 `terminate`는 모두 종료 여부를 의미하지만 파일마다 컬럼명이 다르다.

### 주의점 3
공감 관련 컬럼은 수준이 다르다.
- `listener_behavior`: 청자 반응 요약
- `listener_empathy_tags`: 문서 단위 공감 태그
- `listener_empathy`: 응답 단위 공감 태그

---

## 5. 최종 정리
감정 분석 파트는 연속 대화 파일 2개를 기준으로 진행하고,
RAG/답변 추천 파트는 `rag_documents.csv`, `response_pairs.csv`를 기준으로 진행하면 된다.