# -*- coding: utf-8 -*-
"""
emotion_analyzer.py
===================
발화 단위 감정 분류 모듈.

분류 방식:
  LLM-Based : 구조화 프롬프트를 활용한 정밀 분류 (Chain-of-Thought)

입력:  발화 텍스트 (str) 또는 대화 전체 (list[str])
출력:  구조화된 감정 분석 결과 (dict / list[dict])

참고 기준: docs/analysis_criteria.md
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# ──────────────────────────────────────────────
# 1. 감정 라벨 정의
# ──────────────────────────────────────────────

# 원본 7종 감정 라벨
EMOTION_LABELS = ["중립", "놀람", "분노", "슬픔", "행복", "혐오", "공포"]

EMOTION_LABEL_EN = {
    "중립": "neutral",
    "놀람": "surprise",
    "분노": "anger",
    "슬픔": "sadness",
    "행복": "happiness",
    "혐오": "disgust",
    "공포": "fear",
}

# 상위 3그룹 매핑
EMOTION_GROUP = {
    "분노": "negative",
    "슬픔": "negative",
    "혐오": "negative",
    "공포": "negative",
    "중립": "neutral",
    "행복": "positive",
    "놀람": "positive",
}

# 그룹별 대응 전략
GROUP_STRATEGY = {
    "negative": "공감 표현 + 진정 유도",
    "neutral": "상황 파악 + 부드러운 질문",
    "positive": "긍정 강화 + 해결 유도",
}

# ──────────────────────────────────────────────
# 2. 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class EmotionResult:
    """단일 발화에 대한 감정 분석 결과."""

    utterance: str
    primary: str  # 감정 라벨 (한글)
    primary_en: str  # 감정 라벨 (영문)
    group: str  # 상위 그룹 (negative / neutral / positive)
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    method: str  # 분류 방식 ("llm")
    reasoning: str = ""  # 분류 근거
    strategy: str = ""  # 대응 전략

    @property
    def confidence_percent(self) -> int:
        return int(self.confidence * 100)

    @property
    def confidence_str(self) -> str:
        return f"{self.confidence_percent}%"

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        result = asdict(self)
        result["confidence_percent"] = self.confidence_percent
        result["confidence_str"] = self.confidence_str
        return result

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        """JSON 문자열 변환."""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)


@dataclass
class DialogueEmotionResult:
    """대화 전체에 대한 감정 분석 결과."""

    dialogue_id: Optional[str] = None
    utterance_results: list[EmotionResult] = field(default_factory=list)
    emotion_sequence: list[str] = field(default_factory=list)
    dominant_emotion: str = ""
    dominant_group: str = ""
    negative_ratio: float = 0.0
    emotion_volatility: float = 0.0
    method: str = "llm"

    @property
    def negative_ratio_percent(self) -> int:
        return int(self.negative_ratio * 100)

    @property
    def negative_ratio_str(self) -> str:
        return f"{self.negative_ratio_percent}%"

    @property
    def emotion_volatility_percent(self) -> int:
        return int(self.emotion_volatility * 100)

    @property
    def emotion_volatility_str(self) -> str:
        return f"{self.emotion_volatility_percent}%"

    def to_dict(self) -> dict:
        """딕셔너리 변환."""
        result = asdict(self)
        result["utterance_results"] = [u.to_dict() for u in self.utterance_results]
        result["negative_ratio_percent"] = self.negative_ratio_percent
        result["negative_ratio_str"] = self.negative_ratio_str
        result["emotion_volatility_percent"] = self.emotion_volatility_percent
        result["emotion_volatility_str"] = self.emotion_volatility_str
        return result

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        """JSON 문자열 변환."""
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)


# ──────────────────────────────────────────────
# 3. LLM 기반 감정 분류기
# ──────────────────────────────────────────────

class EmotionClassifier:
    """
    LLM 프롬프트 기반 감정 분류기.

    특징:
      - Chain-of-Thought 추론 방식으로 분류 근거를 단계별 도출
      - JSON 구조화 출력을 강제하여 파싱 안정성 확보
      - 신뢰도(confidence) 자체 평가 포함
      - 실제 LLM API 호출은 외부에서 주입 (llm_caller 함수)

    사용 방식:
      1. classify(utterance, llm_caller) → EmotionResult 반환
      2. classify_dialogue(utterances, llm_caller) → DialogueEmotionResult 반환
    """

    # 단일 발화 분류 프롬프트
    SINGLE_UTTERANCE_PROMPT = """당신은 **연인 갈등 상황** 전문 감정 분석가입니다.

## 과업
아래 발화 텍스트의 **감정 라벨**을 분류하세요.

## 감정 라벨 목록 (7종)
| 라벨 | 영문 | 설명 |
|---|---|---|
| 중립 | neutral | 특별한 감정 없이 사실 전달이나 일상 대화 |
| 놀람 | surprise | 예상치 못한 사실에 대한 놀라움 |
| 분노 | anger | 화남, 짜증, 불만, 공격적 표현 |
| 슬픔 | sadness | 서운함, 우울, 아쉬움, 미안함 |
| 행복 | happiness | 기쁨, 만족, 감사, 긍정적 반응 |
| 혐오 | disgust | 짜증, 경멸, 거부감, 냉소적 반응 |
| 공포 | fear | 불안, 걱정, 두려움, 염려 |

## 분류 규칙
1. 반드시 위 7종 라벨 중 **하나만** 선택
2. 반어법·비꼼이 감지되면 **표면 감정이 아닌 실제 의도된 감정**을 선택
3. 복합 감정이면 **가장 지배적인 감정** 선택
4. 연인 갈등 맥락을 고려하여 판단

## 추론 방식 (Chain-of-Thought)
다음 단계를 반드시 거치세요:
1. [표면 분석] 발화의 표면적 의미 파악
2. [의도 분석] 화자의 실제 의도/감정 추론
3. [맥락 고려] 연인 갈등 상황에서의 맥락 고려
4. [최종 판단] 감정 라벨 확정 + 근거 요약

## 발화 텍스트
\"\"\"{utterance}\"\"\"

## 출력 형식 (반드시 아래 JSON만 출력)
```json
{{
  "primary": "감정라벨(한글)",
  "primary_en": "감정라벨(영문)",
  "group": "negative|neutral|positive",
  "confidence": 0.0~1.0,
  "reasoning": "단계별 추론 요약 (1~3문장)"
}}
```"""

    # 대화 전체 분류 프롬프트
    DIALOGUE_PROMPT = """당신은 **연인 갈등 상황** 전문 감정 분석가입니다.

## 과업
아래 대화의 **각 발화별 감정 라벨**을 분류하고, **대화 전체 감정 흐름**을 분석하세요.

## 감정 라벨 목록 (7종)
중립(neutral), 놀람(surprise), 분노(anger), 슬픔(sadness), 행복(happiness), 혐오(disgust), 공포(fear)

## 상위 그룹
- negative: 분노, 슬픔, 혐오, 공포
- neutral: 중립
- positive: 행복, 놀람

## 분류 규칙
1. 각 발화마다 7종 라벨 중 하나 선택
2. 반어법·비꼼 → 실제 의도된 감정 선택
3. 연인 갈등 맥락 고려

## 대화 내용
{dialogue}

## 출력 형식 (반드시 아래 JSON만 출력)
```json
{{
  "utterances": [
    {{
      "index": 0,
      "text": "발화 텍스트",
      "primary": "감정라벨(한글)",
      "primary_en": "감정라벨(영문)",
      "group": "negative|neutral|positive",
      "confidence": 0.0~1.0,
      "reasoning": "추론 근거 (1문장)"
    }}
  ],
  "dialogue_summary": {{
    "dominant_emotion": "가장 지배적인 감정(한글)",
    "dominant_group": "negative|neutral|positive",
    "emotion_flow": "감정 흐름 설명 (1~2문장)",
    "conflict_level": "low|medium|high"
  }}
}}
```"""

    def get_single_prompt(self, utterance: str) -> str:
        """단일 발화 감정 분류 프롬프트를 생성한다."""
        return self.SINGLE_UTTERANCE_PROMPT.format(utterance=utterance)

    def get_dialogue_prompt(self, utterances: list[str]) -> str:
        """대화 전체 감정 분류 프롬프트를 생성한다."""
        dialogue_text = "\n".join(
            f"[발화 {i}] {u}" for i, u in enumerate(utterances)
        )
        return self.DIALOGUE_PROMPT.format(dialogue=dialogue_text)

    def classify(self, utterance: str, llm_caller) -> EmotionResult:
        """
        단일 발화 감정 분류 (LLM 기반).

        Parameters
        ----------
        utterance : str
            분류 대상 발화 텍스트.
        llm_caller : callable
            LLM 호출 함수. 입력: prompt(str) → 출력: response(str)

        Returns
        -------
        EmotionResult
            구조화된 감정 분석 결과.
        """
        prompt = self.get_single_prompt(utterance)
        response = llm_caller(prompt)
        return self.parse_single_response(utterance, response)

    def classify_dialogue(
        self,
        utterances: list[str],
        llm_caller,
        dialogue_id: str | None = None,
    ) -> DialogueEmotionResult:
        """
        대화(발화 리스트) 전체 감정 분석 (LLM 기반).

        Parameters
        ----------
        utterances : list[str]
            발화 텍스트 리스트.
        llm_caller : callable
            LLM 호출 함수.
        dialogue_id : str | None
            대화 식별자 (선택).

        Returns
        -------
        DialogueEmotionResult
            대화 전체 감정 분석 결과.
        """
        prompt = self.get_dialogue_prompt(utterances)
        response = llm_caller(prompt)
        return self.parse_dialogue_response(utterances, response, dialogue_id)

    def parse_single_response(
        self, utterance: str, llm_output: str
    ) -> EmotionResult:
        """
        LLM 응답을 파싱하여 EmotionResult로 변환한다.

        Parameters
        ----------
        utterance : str
            원본 발화 텍스트.
        llm_output : str
            LLM의 JSON 응답 문자열.

        Returns
        -------
        EmotionResult
            파싱된 감정 분석 결과.

        Raises
        ------
        ValueError
            JSON 파싱 실패 시.
        """
        parsed = self._extract_json(llm_output)
        primary = parsed.get("primary", "중립")
        primary_en = parsed.get("primary_en", EMOTION_LABEL_EN.get(primary, "unknown"))
        group = parsed.get("group", EMOTION_GROUP.get(primary, "neutral"))
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("reasoning", "")

        # 유효성 검증: 알 수 없는 라벨이면 중립 처리
        if primary not in EMOTION_LABELS:
            primary = "중립"
            primary_en = "neutral"
            group = "neutral"
            reasoning = f"[경고] 알 수 없는 라벨 반환 -> 중립 처리. 원본: {parsed.get('primary')}"

        return EmotionResult(
            utterance=utterance,
            primary=primary,
            primary_en=primary_en,
            group=group,
            confidence=confidence,
            method="llm",
            reasoning=reasoning,
            strategy=GROUP_STRATEGY.get(group, ""),
        )

    def parse_dialogue_response(
        self, utterances: list[str], llm_output: str, dialogue_id: str | None = None,
    ) -> DialogueEmotionResult:
        """
        LLM 대화 분석 응답을 파싱하여 DialogueEmotionResult로 변환한다.

        Parameters
        ----------
        utterances : list[str]
            원본 발화 리스트.
        llm_output : str
            LLM의 JSON 응답 문자열.
        dialogue_id : str | None
            대화 식별자.

        Returns
        -------
        DialogueEmotionResult
            대화 전체 감정 분석 결과.
        """
        parsed = self._extract_json(llm_output)
        utt_results_raw = parsed.get("utterances", [])
        summary = parsed.get("dialogue_summary", {})

        utt_results: list[EmotionResult] = []
        for i, raw in enumerate(utt_results_raw):
            text = utterances[i] if i < len(utterances) else raw.get("text", "")
            primary = raw.get("primary", "중립")
            if primary not in EMOTION_LABELS:
                primary = "중립"
            utt_results.append(EmotionResult(
                utterance=text,
                primary=primary,
                primary_en=raw.get("primary_en", EMOTION_LABEL_EN.get(primary, "unknown")),
                group=raw.get("group", EMOTION_GROUP.get(primary, "neutral")),
                confidence=float(raw.get("confidence", 0.5)),
                method="llm",
                reasoning=raw.get("reasoning", ""),
                strategy=GROUP_STRATEGY.get(
                    raw.get("group", EMOTION_GROUP.get(primary, "neutral")), ""
                ),
            ))

        emotion_seq = [r.primary for r in utt_results]
        groups = [r.group for r in utt_results]
        neg_count = sum(1 for g in groups if g == "negative")
        neg_ratio = round(neg_count / len(groups), 4) if groups else 0.0

        if len(emotion_seq) > 1:
            transitions = sum(
                1 for i in range(1, len(emotion_seq))
                if emotion_seq[i] != emotion_seq[i - 1]
            )
            volatility = round(transitions / (len(emotion_seq) - 1), 4)
        else:
            volatility = 0.0

        return DialogueEmotionResult(
            dialogue_id=dialogue_id,
            utterance_results=utt_results,
            emotion_sequence=emotion_seq,
            dominant_emotion=summary.get("dominant_emotion", "중립"),
            dominant_group=summary.get("dominant_group", "neutral"),
            negative_ratio=neg_ratio,
            emotion_volatility=volatility,
            method="llm",
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        LLM 출력에서 JSON 블록을 추출하여 파싱한다.

        코드블록(```json ... ```) 형태와 순수 JSON 모두 처리.
        """
        # 코드블록 내부 JSON 추출 시도
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM 응답 JSON 파싱 실패: {e}\n원본 응답:\n{text[:300]}"
            ) from e


# ──────────────────────────────────────────────
# 4. 통합 분석 함수
# ──────────────────────────────────────────────

def analyze_emotion(
    utterance: str,
    llm_caller=None,
) -> EmotionResult:
    """
    단일 발화 감정 분석 함수 (LLM 기반).

    Parameters
    ----------
    utterance : str
        분석 대상 발화 텍스트.
    llm_caller : callable
        LLM 호출 함수. 필수.
        입력: prompt(str) → 출력: response(str)

    Returns
    -------
    EmotionResult
        구조화된 감정 분석 결과.
    """
    if llm_caller is None:
        raise ValueError("llm_caller 함수를 제공해야 합니다.")

    classifier = EmotionClassifier()
    return classifier.classify(utterance, llm_caller)


def analyze_dialogue_emotion(
    utterances: list[str],
    dialogue_id: str | None = None,
    llm_caller=None,
) -> DialogueEmotionResult:
    """
    대화 전체 감정 분석 함수 (LLM 기반).

    Parameters
    ----------
    utterances : list[str]
        발화 텍스트 리스트.
    dialogue_id : str | None
        대화 식별자.
    llm_caller : callable
        LLM 호출 함수. 필수.

    Returns
    -------
    DialogueEmotionResult
        대화 전체 감정 분석 결과.
    """
    if llm_caller is None:
        raise ValueError("llm_caller 함수를 제공해야 합니다.")

    classifier = EmotionClassifier()
    return classifier.classify_dialogue(utterances, llm_caller, dialogue_id)


# ──────────────────────────────────────────────
# 5. CLI 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # ── Mock LLM Caller (테스트용) ──
    # 실제 LLM 없이 테스트하기 위한 더미 응답 생성기
    def mock_llm_caller(prompt: str) -> str:
        """테스트용 Mock LLM 응답 생성."""
        # 발화 텍스트에서 간단히 감정 추정 (테스트 목적)
        if "화나" in prompt or "짜증" in prompt or "미치겠" in prompt or "피지 말라" in prompt:
            return json.dumps({
                "primary": "분노", "primary_en": "anger", "group": "negative",
                "confidence": 0.85, "reasoning": "공격적 어투와 명령형 표현에서 분노 감정이 드러남"
            }, ensure_ascii=False)
        elif "걱정" in prompt or "불안" in prompt or "무서" in prompt:
            return json.dumps({
                "primary": "공포", "primary_en": "fear", "group": "negative",
                "confidence": 0.7, "reasoning": "걱정과 불안 표현에서 공포/염려 감정 확인"
            }, ensure_ascii=False)
        elif "좋아" in prompt or "대박" in prompt or "기뻐" in prompt:
            return json.dumps({
                "primary": "행복", "primary_en": "happiness", "group": "positive",
                "confidence": 0.9, "reasoning": "긍정적 감탄과 기쁨 표현"
            }, ensure_ascii=False)
        elif "지긋지긋" in prompt or "재수없" in prompt:
            return json.dumps({
                "primary": "혐오", "primary_en": "disgust", "group": "negative",
                "confidence": 0.8, "reasoning": "경멸과 거부 표현에서 혐오 감정 확인"
            }, ensure_ascii=False)
        elif "미안" in prompt or "잘못" in prompt or "서운" in prompt:
            return json.dumps({
                "primary": "슬픔", "primary_en": "sadness", "group": "negative",
                "confidence": 0.75, "reasoning": "미안함과 자기 비난에서 슬픔 감정 확인"
            }, ensure_ascii=False)
        elif "손님" in prompt:
            return json.dumps({
                "primary": "중립", "primary_en": "neutral", "group": "neutral",
                "confidence": 0.9, "reasoning": "사실 전달의 일상 대화"
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "primary": "중립", "primary_en": "neutral", "group": "neutral",
                "confidence": 0.6, "reasoning": "특별한 감정 표현 없음"
            }, ensure_ascii=False)

    def mock_dialogue_llm_caller(prompt: str) -> str:
        """테스트용 Mock 대화 분석 LLM 응답 생성."""
        return json.dumps({
            "utterances": [
                {"index": 0, "text": "", "primary": "분노", "primary_en": "anger",
                 "group": "negative", "confidence": 0.9, "reasoning": "강한 불만 표출"},
                {"index": 1, "text": "", "primary": "중립", "primary_en": "neutral",
                 "group": "neutral", "confidence": 0.7, "reasoning": "무덤덤한 제안"},
                {"index": 2, "text": "", "primary": "분노", "primary_en": "anger",
                 "group": "negative", "confidence": 0.85, "reasoning": "금전 불만"},
                {"index": 3, "text": "", "primary": "슬픔", "primary_en": "sadness",
                 "group": "negative", "confidence": 0.6, "reasoning": "체념 표현"},
                {"index": 4, "text": "", "primary": "슬픔", "primary_en": "sadness",
                 "group": "negative", "confidence": 0.8, "reasoning": "속상함 토로"},
            ],
            "dialogue_summary": {
                "dominant_emotion": "분노",
                "dominant_group": "negative",
                "emotion_flow": "분노 → 중립 → 분노 → 슬픔 → 슬픔으로 감정이 악화됨",
                "conflict_level": "high"
            }
        }, ensure_ascii=False)

    print("=" * 60)
    print("감정 분석기 테스트 (LLM-Based)")
    print("=" * 60)

    test_utterances = [
        "아 진짜! 사무실에서 피지 말라니깐! 간접흡연이 얼마나 안좋은데!",
        "손님 왔어요.",
        "난 그냥... 걱정 돼서...",
        "대박! 진짜? 나 너무 좋아!",
        "지긋지긋해. 재수없어.",
        "뭔가 말리는 기분이야. 불안해.",
        "알았어. 내가 잘못했어. 미안해.",
    ]

    clf = EmotionClassifier()

    for utt in test_utterances:
        result = clf.classify(utt, mock_llm_caller)
        print(f"\n발화: {utt}")
        print(f"  감정: {result.primary} ({result.primary_en})")
        print(f"  그룹: {result.group}")
        print(f"  신뢰도: {result.confidence_str}")
        print(f"  전략: {result.strategy}")
        print(f"  근거: {result.reasoning}")

    # 대화 전체 분석 테스트
    print("\n" + "=" * 60)
    print("대화 전체 감정 분석 테스트")
    print("=" * 60)

    dialogue = [
        "너 어떻게 된 거야! 한 시간두 넘게 기다렸잖아!",
        "그냥 열쇠 집 불러서 열지.",
        "그런데 쓸 돈이 어딨어? 돈이 남아돌아?",
        "알았어. 그만 해.",
        "오늘도 2만원 밖에 못 팔고 들어와서 속상해 죽겠는데!",
    ]

    dial_result = clf.classify_dialogue(dialogue, mock_dialogue_llm_caller, dialogue_id="test_001")
    print(f"\n감정 시퀀스: {' > '.join(dial_result.emotion_sequence)}")
    print(f"지배적 감정: {dial_result.dominant_emotion} ({dial_result.dominant_group})")
    print(f"부정 감정 비율: {dial_result.negative_ratio_str}")
    print(f"감정 변동성: {dial_result.emotion_volatility_str}")

    # LLM 프롬프트 예시 출력
    print("\n" + "=" * 60)
    print("LLM 프롬프트 예시")
    print("=" * 60)
    prompt = clf.get_single_prompt("왜 화를 내? 그냥 자주 왔다 갔다 하면 되잖아.")
    print(prompt[:500] + "...")
