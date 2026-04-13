# -*- coding: utf-8 -*-
"""
emotion_analyzer.py
===================
발화 단위 감정 분류 모듈.

분류 방식:
  1. Rule-Based: 키워드 매칭 + 감정 패턴 기반 빠른 분류
  2. LLM-Based : 구조화 프롬프트를 활용한 정밀 분류 (Chain-of-Thought)

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
# 2. 감정별 키워드 사전 (Rule-Based)
# ──────────────────────────────────────────────

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "분노": [
        "화나", "화났", "짜증", "열받", "빡치", "빡쳐", "미치겠",
        "환장", "꺼져", "닥쳐", "진짜", "어떻게", "왜이래", "왜그래",
        "말라니깐", "말랬지", "하지마", "그만해", "됐거든", "짜증나",
        "구박", "속상", "못참", "화를", "버릇이", "뻔뻔",
    ],
    "슬픔": [
        "서운", "섭섭", "슬퍼", "슬프", "울고", "울었", "눈물",
        "미안", "외로", "혼자", "그리워", "그립", "아프", "힘들",
        "답답", "우울", "속상", "걱정", "안쓰럽", "불쌍",
        "후회", "잘못했", "죄송", "괴로",
    ],
    "행복": [
        "좋아", "좋겠", "행복", "기뻐", "기쁘", "신나", "웃기",
        "재밌", "재미있", "고마워", "고맙", "감사", "사랑",
        "최고", "짱", "대박", "축하", "멋지", "예쁘",
        "다행", "즐거", "설레", "반가",
    ],
    "놀람": [
        "헐", "세상에", "진짜?", "정말?", "말도안돼", "어머",
        "깜짝", "놀랐", "놀라", "대박", "미쳤", "설마",
        "에이", "아니", "뭐라고", "거짓말", "놀래",
    ],
    "혐오": [
        "역겹", "징그럽", "지겹", "지긋지긋", "재수없", "싸가지",
        "한심", "찌질", "못났", "쓸모없", "경멸", "꼴",
        "더럽", "치사", "비열", "꼴불견", "못생",
    ],
    "공포": [
        "무서", "두려", "겁나", "불안", "걱정", "무섭",
        "불길", "위험", "떨려", "놀랐", "공포", "겁",
        "조심", "위협", "피해",
    ],
}

# ──────────────────────────────────────────────
# 3. 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class EmotionResult:
    """단일 발화에 대한 감정 분석 결과."""

    utterance: str
    primary: str  # 감정 라벨 (한글)
    primary_en: str  # 감정 라벨 (영문)
    group: str  # 상위 그룹 (negative / neutral / positive)
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    method: str  # 분류 방식 ("rule_based" | "llm")
    reasoning: str = ""  # 분류 근거 (LLM 사용 시)
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
    method: str = "rule_based"

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
# 4. Rule-Based 감정 분류기
# ──────────────────────────────────────────────

class RuleBasedEmotionClassifier:
    """
    키워드 매칭 기반 감정 분류기.

    동작 원리:
      1. 입력 발화에서 각 감정별 키워드 매칭 횟수를 계산
      2. 가장 많이 매칭된 감정을 1차 후보로 선정
      3. 매칭이 없으면 '중립'으로 분류
      4. 신뢰도는 (최다 매칭 수) / (전체 매칭 수) 로 산출
    """

    def __init__(self) -> None:
        self.keywords = EMOTION_KEYWORDS
        # 키워드를 정규식 패턴으로 사전 컴파일
        self._patterns: dict[str, list[re.Pattern]] = {}
        for emotion, kw_list in self.keywords.items():
            self._patterns[emotion] = [
                re.compile(re.escape(kw)) for kw in kw_list
            ]

    def classify(self, utterance: str) -> EmotionResult:
        """
        단일 발화 감정 분류.

        Parameters
        ----------
        utterance : str
            분류 대상 발화 텍스트.

        Returns
        -------
        EmotionResult
            구조화된 감정 분석 결과.
        """
        text = utterance.strip()
        if not text:
            return self._make_result(text, "중립", 0.5, "빈 발화 -> 중립 처리")

        # 각 감정별 키워드 매칭 횟수 계산
        scores: dict[str, int] = {}
        for emotion, patterns in self._patterns.items():
            count = sum(1 for p in patterns if p.search(text))
            if count > 0:
                scores[emotion] = count

        if not scores:
            return self._make_result(text, "중립", 0.5, "키워드 매칭 없음 -> 중립 처리")

        # 최다 매칭 감정 선정
        total = sum(scores.values())
        best_emotion = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_count = scores[best_emotion]
        confidence = round(best_count / total, 2) if total > 0 else 0.5

        # 신뢰도 하한 보정: 단일 매칭은 최소 0.4
        confidence = max(confidence, 0.4)

        reasoning = (
            f"키워드 매칭 결과 - "
            + ", ".join(f"{e}: {c}회" for e, c in sorted(scores.items(), key=lambda x: -x[1]))
            + f" -> '{best_emotion}' 선정 (신뢰도 {confidence})"
        )

        return self._make_result(text, best_emotion, confidence, reasoning)

    def classify_dialogue(
        self,
        utterances: list[str],
        dialogue_id: str | None = None,
    ) -> DialogueEmotionResult:
        """
        대화(발화 리스트) 전체 감정 분석.

        Parameters
        ----------
        utterances : list[str]
            발화 텍스트 리스트.
        dialogue_id : str | None
            대화 식별자 (선택).

        Returns
        -------
        DialogueEmotionResult
            대화 전체 감정 분석 결과.
        """
        results = [self.classify(u) for u in utterances]
        emotion_seq = [r.primary for r in results]
        groups = [r.group for r in results]

        # 지배적 감정: 중립 제외 최빈 감정 (전부 중립이면 중립)
        non_neutral = [e for e in emotion_seq if e != "중립"]
        if non_neutral:
            from collections import Counter
            dominant = Counter(non_neutral).most_common(1)[0][0]
        else:
            dominant = "중립"

        # 부정 감정 비율
        neg_count = sum(1 for g in groups if g == "negative")
        neg_ratio = round(neg_count / len(groups), 4) if groups else 0.0

        # 감정 변동성: 감정 전환 횟수 / (전체 - 1)
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
            utterance_results=results,
            emotion_sequence=emotion_seq,
            dominant_emotion=dominant,
            dominant_group=EMOTION_GROUP.get(dominant, "neutral"),
            negative_ratio=neg_ratio,
            emotion_volatility=volatility,
            method="rule_based",
        )

    @staticmethod
    def _make_result(
        utterance: str,
        emotion: str,
        confidence: float,
        reasoning: str,
    ) -> EmotionResult:
        """EmotionResult 생성 헬퍼."""
        return EmotionResult(
            utterance=utterance,
            primary=emotion,
            primary_en=EMOTION_LABEL_EN.get(emotion, "unknown"),
            group=EMOTION_GROUP.get(emotion, "neutral"),
            confidence=confidence,
            method="rule_based",
            reasoning=reasoning,
            strategy=GROUP_STRATEGY.get(EMOTION_GROUP.get(emotion, "neutral"), ""),
        )


# ──────────────────────────────────────────────
# 5. LLM 기반 감정 분류기
# ──────────────────────────────────────────────

class LLMEmotionClassifier:
    """
    LLM 프롬프트 기반 감정 분류기.

    특징:
      - Chain-of-Thought 추론 방식으로 분류 근거를 단계별 도출
      - JSON 구조화 출력을 강제하여 파싱 안정성 확보
      - 신뢰도(confidence) 자체 평가 포함
      - 실제 LLM API 호출은 외부에서 주입 (call_llm 함수)

    사용 방식:
      1. get_prompt() → 프롬프트 문자열 생성
      2. 외부에서 LLM API 호출
      3. parse_response() → 응답 파싱 → EmotionResult 반환
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
# 6. 통합 분석 함수
# ──────────────────────────────────────────────

def analyze_emotion(
    utterance: str,
    method: str = "rule_based",
    llm_caller=None,
) -> EmotionResult:
    """
    단일 발화 감정 분석 통합 함수.

    Parameters
    ----------
    utterance : str
        분석 대상 발화 텍스트.
    method : str
        분류 방식. "rule_based" 또는 "llm".
    llm_caller : callable | None
        LLM 호출 함수. method="llm" 시 필수.
        입력: prompt(str) → 출력: response(str)

    Returns
    -------
    EmotionResult
        구조화된 감정 분석 결과.
    """
    if method == "rule_based":
        classifier = RuleBasedEmotionClassifier()
        return classifier.classify(utterance)

    elif method == "llm":
        if llm_caller is None:
            raise ValueError("LLM 방식 사용 시 llm_caller 함수를 제공해야 합니다.")
        llm_clf = LLMEmotionClassifier()
        prompt = llm_clf.get_single_prompt(utterance)
        response = llm_caller(prompt)
        return llm_clf.parse_single_response(utterance, response)

    else:
        raise ValueError(f"지원하지 않는 분류 방식: {method}. 'rule_based' 또는 'llm'을 사용하세요.")


def analyze_dialogue_emotion(
    utterances: list[str],
    dialogue_id: str | None = None,
    method: str = "rule_based",
    llm_caller=None,
) -> DialogueEmotionResult:
    """
    대화 전체 감정 분석 통합 함수.

    Parameters
    ----------
    utterances : list[str]
        발화 텍스트 리스트.
    dialogue_id : str | None
        대화 식별자.
    method : str
        분류 방식. "rule_based" 또는 "llm".
    llm_caller : callable | None
        LLM 호출 함수. method="llm" 시 필수.

    Returns
    -------
    DialogueEmotionResult
        대화 전체 감정 분석 결과.
    """
    if method == "rule_based":
        classifier = RuleBasedEmotionClassifier()
        return classifier.classify_dialogue(utterances, dialogue_id)

    elif method == "llm":
        if llm_caller is None:
            raise ValueError("LLM 방식 사용 시 llm_caller 함수를 제공해야 합니다.")
        llm_clf = LLMEmotionClassifier()
        prompt = llm_clf.get_dialogue_prompt(utterances)
        response = llm_caller(prompt)
        return llm_clf.parse_dialogue_response(utterances, response, dialogue_id)

    else:
        raise ValueError(f"지원하지 않는 분류 방식: {method}")


# ──────────────────────────────────────────────
# 7. 교차 검증 유틸리티
# ──────────────────────────────────────────────

def cross_validate(
    utterance: str,
    llm_caller=None,
) -> dict:
    """
    Rule-Based와 LLM 결과를 교차 검증한다.

    Parameters
    ----------
    utterance : str
        분석 대상 발화 텍스트.
    llm_caller : callable
        LLM 호출 함수.

    Returns
    -------
    dict
        교차 검증 결과.
        - rule_based: Rule-Based 결과
        - llm: LLM 결과
        - match: 두 결과 일치 여부
        - final: 최종 채택 결과
        - conflict_note: 불일치 시 설명
    """
    rule_result = analyze_emotion(utterance, method="rule_based")

    if llm_caller is None:
        return {
                "rule_based": rule_result.to_dict(),
            "llm": None,
            "match": None,
            "final": rule_result.to_dict(),
            "conflict_note": "LLM 미사용 - Rule-Based 결과 단독 채택",
        }

    llm_result = analyze_emotion(utterance, method="llm", llm_caller=llm_caller)
    is_match = rule_result.primary == llm_result.primary

    # 최종 채택 로직: LLM 신뢰도가 0.6 이상이면 LLM 우선, 아니면 Rule-Based
    if is_match:
        final = llm_result
        note = "Rule-Based와 LLM 결과 일치 -> LLM 결과 채택"
    elif llm_result.confidence >= 0.6:
        final = llm_result
        note = (
            f"불일치 발생 (Rule: {rule_result.primary}, LLM: {llm_result.primary}). "
            f"LLM 신뢰도({llm_result.confidence}) >= 0.6 -> LLM 결과 채택"
        )
    else:
        final = rule_result
        note = (
            f"불일치 발생 (Rule: {rule_result.primary}, LLM: {llm_result.primary}). "
            f"LLM 신뢰도({llm_result.confidence}) < 0.6 -> Rule-Based 결과 채택"
        )

    return {
        "rule_based": rule_result.to_dict(),
        "llm": llm_result.to_dict(),
        "match": is_match,
        "final": final.to_dict(),
        "conflict_note": note,
    }


# ──────────────────────────────────────────────
# 8. CLI 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("감정 분석기 테스트 (Rule-Based)")
    print("=" * 60)

    test_utterances = [
        "아 진짜! 사무실에서 피지 말라니깐! 간접흡연이 얼마나 안좋은데!",
        "손님 왔어요.",
        "난 그냥... 걱정 돼서...",
        "대박! 진짜? 나 너무 좋아!",
        "지긋지긋해. 재수없어.",
        "뭔가 말리는 기분이야. 불길해.",
        "알았어. 내가 잘못했어. 미안해.",
    ]

    clf = RuleBasedEmotionClassifier()

    for utt in test_utterances:
        result = clf.classify(utt)
        print(f"\n발화: {utt}")
        print(f"  감정: {result.primary} ({result.primary_en})")
        print(f"  그룹: {result.group}")
        print(f"  신뢰도: {result.confidence}")
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

    dial_result = clf.classify_dialogue(dialogue, dialogue_id="test_001")
    print(f"\n감정 시퀀스: {' > '.join(dial_result.emotion_sequence)}")
    print(f"지배적 감정: {dial_result.dominant_emotion} ({dial_result.dominant_group})")
    print(f"부정 감정 비율: {dial_result.negative_ratio}")
    print(f"감정 변동성: {dial_result.emotion_volatility}")

    # LLM 프롬프트 예시 출력
    print("\n" + "=" * 60)
    print("LLM 프롬프트 예시")
    print("=" * 60)
    llm_clf = LLMEmotionClassifier()
    prompt = llm_clf.get_single_prompt("왜 화를 내? 그냥 자주 왔다 갔다 하면 되잖아.")
    print(prompt[:500] + "...")
