# Issue: rule-based 분석 제거 및 LLM-based 감정/위험도 분석 구조 전환

## 배경

기존 감정 분석 및 갈등 위험도 분석은 키워드/규칙 기반 판단에 의존하는 구조였다.
이 방식은 구현이 단순하고 빠르지만, 연인 갈등 대화처럼 반어법, 비꼼, 복합 감정, 문맥 전환이 많은 입력에서는 실제 감정과 위험도를 안정적으로 판별하기 어렵다.

이에 따라 `emotion_analyzer.py`, `risk_analyzer.py`를 중심으로 rule-based 판단 로직을 제거하고, 구조화 프롬프트를 사용하는 LLM-based 분석 구조로 전환한다.

---

## 작업 내용

### 1. 감정 분석 구조 변경

대상 파일:

- `src/emotion/emotion_analyzer.py`

변경 내용:

- 단일 발화 감정 분석을 LLM 기반으로 수행한다.
- 대화 전체 감정 흐름 분석을 LLM 기반으로 수행한다.
- 감정 라벨은 기존 7종 라벨 체계를 유지한다.
  - `중립`
  - `놀람`
  - `분노`
  - `슬픔`
  - `행복`
  - `혐오`
  - `공포`
- 감정 라벨별 영문명과 상위 그룹 매핑은 유지한다.
  - `negative`: 분노, 슬픔, 혐오, 공포
  - `neutral`: 중립
  - `positive`: 행복, 놀람
- LLM 응답은 JSON 형식으로 강제하고, 결과를 dataclass로 변환한다.
- 실제 LLM API 호출은 모듈 내부에 고정하지 않고 `llm_caller` 함수로 외부에서 주입한다.

주요 구조:

- `EmotionResult`: 단일 발화 감정 분석 결과
- `DialogueEmotionResult`: 대화 전체 감정 흐름 분석 결과
- `EmotionClassifier`: LLM 프롬프트 생성, 호출, 응답 파싱 담당
- `analyze_emotion()`: 단일 발화 분석용 외부 함수
- `analyze_dialogue_emotion()`: 대화 전체 분석용 외부 함수

---

### 2. 위험도 분석 구조 변경

대상 파일:

- `src/emotion/risk_analyzer.py`

변경 내용:

- 갈등 위험도를 LLM 기반으로 판단한다.
- 기존 위험도 5단계 체계는 유지한다.
  - `safe`: 안전
  - `caution`: 주의
  - `warning`: 경고
  - `danger`: 위험
  - `critical`: 심각
- 위험도 분석 시 감정 분석 결과의 `emotion_sequence`를 참고한다.
- 감정 분석 결과가 없으면 내부에서 `EmotionClassifier`를 사용해 먼저 감정 분석을 수행한다.
- LLM 응답은 JSON 형식으로 강제하고, `RiskResult`로 변환한다.
- 위험도 점수, 위험도 등급, 한글 라벨, 대응 전략, 판단 근거를 함께 반환한다.

주요 구조:

- `RiskResult`: 대화 단위 위험도 분석 결과
- `RiskAnalyzer`: LLM 프롬프트 생성, 호출, 응답 파싱 담당
- `analyze_risk()`: 위험도 분석용 외부 함수
- `full_analysis()`: 감정 분석 후 위험도 분석까지 연결하는 통합 함수

---

## 이전 구조와 변경점

### Before: rule-based 중심

- 키워드 매칭과 규칙 기반 점수 계산에 의존했다.
- 특정 단어가 있으면 감정 또는 위험도를 고정적으로 추정했다.
- 빠르고 비용이 낮지만, 다음 상황에서 한계가 있었다.
  - 반어법
  - 비꼼
  - 복합 감정
  - 문맥상 의미 변화
  - 관계 맥락에 따른 위험도 차이
  - 같은 단어라도 상황별로 다른 감정 표현

### After: LLM-based 중심

- 감정 분석과 위험도 분석 모두 LLM 프롬프트 기반으로 변경한다.
- Chain-of-Thought 형식의 분석 관점을 프롬프트에 포함한다.
- 출력은 JSON으로 고정해 후속 로직에서 구조화된 결과를 사용할 수 있게 한다.
- LLM 호출부는 `llm_caller`로 외부 주입하여 OpenAI, Gemini, Ollama 등 모델 교체가 가능하게 한다.
- 분석 결과에는 라벨뿐 아니라 신뢰도, 추론 근거, 대응 전략을 포함한다.

---

## 코드 구조

```text
src/emotion/
  __init__.py
    - emotion 패키지 외부 공개 API 정리

  emotion_analyzer.py
    - 감정 라벨 정의
    - 감정 그룹 및 대응 전략 정의
    - EmotionResult
    - DialogueEmotionResult
    - EmotionClassifier
    - analyze_emotion()
    - analyze_dialogue_emotion()

  risk_analyzer.py
    - 위험도 5단계 정의
    - 위험도별 대응 전략 정의
    - RiskResult
    - RiskAnalyzer
    - analyze_risk()
    - full_analysis()
```

---

## 기대 효과

- 연인 갈등 대화의 문맥을 더 잘 반영할 수 있다.
- 단순 키워드 기반 오판을 줄일 수 있다.
- 감정 분석 결과와 위험도 분석 결과를 하나의 파이프라인으로 연결할 수 있다.
- RAG/Streamlit/챗봇 응답 생성 단계에서 구조화된 분석 결과를 재사용할 수 있다.
- 모델 호출부가 외부 주입 방식이므로 배포 환경별 LLM provider 교체가 쉽다.

---

## 검증 기준

- `analyze_emotion()` 호출 시 `llm_caller`가 없으면 명확한 `ValueError`가 발생해야 한다.
- `analyze_dialogue_emotion()` 호출 시 발화별 감정 결과와 전체 감정 흐름이 반환되어야 한다.
- `analyze_risk()` 호출 시 위험도 점수, 등급, 라벨, 대응 전략이 반환되어야 한다.
- `full_analysis()` 호출 시 감정 분석 결과와 위험도 분석 결과가 함께 반환되어야 한다.
- LLM 응답이 JSON code block 안에 있거나 순수 JSON 문자열이어도 파싱되어야 한다.
- 파싱 실패 시 원본 응답 일부를 포함한 오류 메시지가 발생해야 한다.

---

## 남은 과제

- 실제 OpenAI/Gemini/Ollama caller 연결부를 별도 모듈로 정리한다.
- Streamlit 배포 환경에서는 `.streamlit/secrets.toml` 또는 Streamlit Cloud Secrets를 통해 API key를 주입한다.
- LLM 응답 JSON schema를 더 엄격하게 검증한다.
- 고위험 입력(`danger`, `critical`)에 대한 안전 응답 정책을 RAG/응답 생성 단계와 연결한다.
- 기존 benchmark와 연결해 rule-based 대비 LLM-based 결과 품질을 비교한다.
