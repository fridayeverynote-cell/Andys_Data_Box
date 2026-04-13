# -*- coding: utf-8 -*-
"""
risk_analyzer.py
================
대화 단위 갈등 위험도 분석 모듈.

분류 방식:
  1. Rule-Based: 감정 시퀀스 통계 + 위험 키워드 매칭 기반 점수 산출
  2. LLM-Based : 구조화 프롬프트를 활용한 정밀 위험도 판정 (Chain-of-Thought)

입력:  대화 발화 리스트 (list[str]) + 감정 분석 결과 (DialogueEmotionResult)
출력:  구조화된 위험도 분석 결과 (dict)

위험도 5단계:
  1. 안전(safe)       0.0~0.2  — 갈등 징후 없음
  2. 주의(caution)    0.2~0.4  — 경미한 불만
  3. 경고(warning)    0.4~0.6  — 명확한 갈등 신호
  4. 위험(danger)     0.6~0.8  — 강한 갈등
  5. 심각(critical)   0.8~1.0  — 극단적 갈등

참고 기준: docs/analysis_criteria.md
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from .emotion_analyzer import (
    EMOTION_GROUP,
    EMOTION_LABEL_EN,
    DialogueEmotionResult,
    RuleBasedEmotionClassifier,
    LLMEmotionClassifier,
)

# ──────────────────────────────────────────────
# 1. 위험도 등급 정의
# ──────────────────────────────────────────────

RISK_LEVELS = [
    {"level": 1, "label": "안전", "label_en": "safe",     "min": 0.0, "max": 0.2},
    {"level": 2, "label": "주의", "label_en": "caution",  "min": 0.2, "max": 0.4},
    {"level": 3, "label": "경고", "label_en": "warning",  "min": 0.4, "max": 0.6},
    {"level": 4, "label": "위험", "label_en": "danger",   "min": 0.6, "max": 0.8},
    {"level": 5, "label": "심각", "label_en": "critical", "min": 0.8, "max": 1.0},
]

RISK_RECOMMENDATIONS = {
    "safe":     "일반 응답",
    "caution":  "공감 표현 + 상황 확인",
    "warning":  "적극적 공감 + 진정 유도",
    "danger":   "긴급 공감 + 대화 전환 시도",
    "critical": "전문 상담 권유 + 안전 대응",
}

# ──────────────────────────────────────────────
# 2. 위험 키워드 사전 (5등급)
# ──────────────────────────────────────────────

# ── 등급 1: 극위험 (가중치 1.0) ──
# 폭력·자해·자살 암시, 극단적 행위 위협
# 즉각적인 안전 개입이 필요한 표현
CRITICAL_RISK_KEYWORDS = [
    # 자해/자살 암시
    "죽겠다", "죽고싶", "죽을래", "죽어버릴", "살기싫", "못살겠",
    "목숨", "유서",
    # 폭력 위협
    "때리", "때릴", "패버릴", "죽여버릴", "없애버릴", "찔러",
    "칼", "폭력",
    # 극단적 행동 예고
    "뛰어내릴", "끝장", "다죽자", "같이죽자",
]

# ── 등급 2: 고위험 (가중치 0.8) ──
# 관계 단절 선언, 강한 거부·추방 표현
# 관계의 존속 자체를 위협하는 표현
HIGH_RISK_KEYWORDS = [
    # 관계 단절 선언
    "헤어지자", "이별", "끝내자", "끝이야", "끝났어", "그만하자",
    "이혼", "갈라서자", "나가살아", "각자살자",
    # 강한 거부·추방
    "꺼져", "사라져", "보기싫어", "지긋지긋", "질렸어",
    "상종", "인연끊", "연락하지마", "다신보지말자",
    # 결정적 포기
    "포기", "체념", "미련없",
]

# ── 등급 3: 중위험 (가중치 0.6) ──
# 인격 비난, 경멸, 모욕성 발언
# 상대방 자존감을 직접 공격하는 표현
MEDIUM_RISK_KEYWORDS = [
    # 인격 비하·모욕
    "한심", "찌질", "못났", "쓸모없", "무능",
    "멍청", "바보같", "저질", "수준이하", "한심하기",
    # 경멸·혐오
    "싸가지", "재수없", "꼴불견", "역겹", "구역질",
    "못생", "징그럽", "더럽",
    # 비열·도덕적 비난
    "비열", "치사", "뻔뻔", "파렴치", "양심없",
    "이기적", "매너없", "인간이하", "쓰레기",
    # 신뢰 파괴
    "배신", "배신자",
]

# ── 등급 4: 주의 (가중치 0.4) ──
# 강한 불만·짜증, 불신 표현, 공격적 감정 분출
# 갈등이 명확하나 관계 단절까지는 아닌 표현
ELEVATED_RISK_KEYWORDS = [
    # 강한 감정 분출
    "짜증", "열받", "미치겠", "환장", "빡치", "빡쳐",
    "화나", "화났", "폭발", "참을수없",
    # 불신·의심
    "거짓말", "뻥", "속이", "거짓", "믿을수없",
    "의심", "수상", "숨기", "감추",
    # 공격적 명령·비난
    "닥쳐", "입다물", "나가", "말걸지마",
    "구박", "못참", "안돼", "안져",
    # 극단적 부정 감정 표출
    "못견디겠", "견딜수없", "미칠것같", "돌아버리겠",
]

# ── 등급 5: 저위험 (가중치 0.2) ──
# 경미한 서운함, 실망, 소극적 불만
# 갈등 초기 신호이나 직접적 공격성은 없는 표현
LOW_RISK_KEYWORDS = [
    # 서운함·실망
    "서운", "섭섭", "실망", "아쉽",
    # 답답함·속상함
    "답답", "속상", "걱정", "불안",
    "우울", "힘들", "지치",
    # 소극적 불만
    "왜그래", "왜이래", "그러지마", "하지마",
    "그만해", "됐어", "알았어", "몰라",
    # 회피·냉담
    "말하기싫", "귀찮", "상관없", "알아서해",
]

KEYWORD_TIERS = [
    {"keywords": CRITICAL_RISK_KEYWORDS, "weight": 1.0, "severity": "critical"},
    {"keywords": HIGH_RISK_KEYWORDS,     "weight": 0.8, "severity": "high"},
    {"keywords": MEDIUM_RISK_KEYWORDS,   "weight": 0.6, "severity": "medium"},
    {"keywords": ELEVATED_RISK_KEYWORDS, "weight": 0.4, "severity": "elevated"},
    {"keywords": LOW_RISK_KEYWORDS,      "weight": 0.2, "severity": "low"},
]

# ──────────────────────────────────────────────
# 3. 시그널 가중치
# ──────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "negative_ratio": 0.25,   # 부정 감정 비율
    "anger_ratio":    0.20,   # 분노 집중도
    "keyword_score":  0.25,   # 위험 키워드 탐지 점수
    "streak_score":   0.15,   # 부정 감정 연속성
    "ending_score":   0.15,   # 대화 종료 감정
}

# ──────────────────────────────────────────────
# 4. 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class DetectedKeyword:
    """탐지된 위험 키워드."""
    keyword: str
    severity: str  # critical / high / medium / elevated / low
    weight: float
    utterance_index: int = -1  # 발생 위치

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskResult:
    """대화 단위 위험도 분석 결과."""

    dialogue_id: Optional[str] = None
    risk_score: float = 0.0               # 최종 위험도 점수 (0.0~1.0)
    risk_level: str = "safe"              # 위험도 영문 라벨
    risk_label: str = "안전"               # 위험도 한글 라벨
    risk_grade: int = 1                   # 위험도 등급 (1~5)
    signals: dict = field(default_factory=dict)         # 세부 시그널
    detected_keywords: list[DetectedKeyword] = field(default_factory=list)
    emotion_sequence: list[str] = field(default_factory=list)
    recommendation: str = ""              # 대응 전략
    method: str = "rule_based"
    reasoning: str = ""                   # LLM 사용 시 추론 근거

    @property
    def risk_score_percent(self) -> int:
        """100분율로 환산된 위험도 점수"""
        return int(self.risk_score * 100)

    @property
    def risk_score_str(self) -> str:
        """100% 표기법 문자열"""
        return f"{self.risk_score_percent}%"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["risk_score_percent"] = self.risk_score_percent
        result["risk_score_str"] = self.risk_score_str

        # 내부 시그널(세부 분석 지표)들도 퍼센트 표기 추가
        if "signals" in result and isinstance(result["signals"], dict):
            result["signals_percent"] = {}
            for k, v in list(result["signals"].items()):
                if isinstance(v, float):
                    result["signals_percent"][f"{k}_percent"] = int(v * 100)
                    result["signals_percent"][f"{k}_str"] = f"{int(v * 100)}%"

        return result

    def to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)


# ──────────────────────────────────────────────
# 5. Rule-Based 위험도 분석기
# ──────────────────────────────────────────────

class RuleBasedRiskAnalyzer:
    """
    감정 시퀀스 통계 + 위험 키워드 매칭 기반 위험도 분석기.

    동작 원리:
      1. 감정 분석 결과에서 통계 시그널 추출
      2. 발화 텍스트에서 위험 키워드 탐지
      3. 5개 시그널의 가중 합산으로 최종 점수 산출
      4. 점수 구간에 따라 위험도 등급 결정
    """

    def __init__(self) -> None:
        # 키워드 패턴 사전 컴파일
        self._keyword_patterns: list[dict] = []
        for tier in KEYWORD_TIERS:
            for kw in tier["keywords"]:
                self._keyword_patterns.append({
                    "keyword": kw,
                    "pattern": re.compile(re.escape(kw)),
                    "weight": tier["weight"],
                    "severity": tier["severity"],
                })

    def analyze(
        self,
        utterances: list[str],
        emotion_result: DialogueEmotionResult | None = None,
        dialogue_id: str | None = None,
    ) -> RiskResult:
        """
        대화 위험도 분석.

        Parameters
        ----------
        utterances : list[str]
            대화 발화 리스트.
        emotion_result : DialogueEmotionResult | None
            사전 수행된 감정 분석 결과. None이면 내부에서 감정 분석 수행.
        dialogue_id : str | None
            대화 식별자.

        Returns
        -------
        RiskResult
            구조화된 위험도 분석 결과.
        """
        # 감정 분석이 없으면 수행
        if emotion_result is None:
            clf = RuleBasedEmotionClassifier()
            emotion_result = clf.classify_dialogue(utterances, dialogue_id)

        emotion_seq = emotion_result.emotion_sequence
        groups = [EMOTION_GROUP.get(e, "neutral") for e in emotion_seq]

        # ── 시그널 1: 부정 감정 비율 ──
        neg_ratio = emotion_result.negative_ratio

        # ── 시그널 2: 분노 집중도 (분노 + 혐오) ──
        anger_disgust = sum(
            1 for e in emotion_seq if e in ("분노", "혐오")
        )
        anger_ratio = round(anger_disgust / len(emotion_seq), 4) if emotion_seq else 0.0

        # ── 시그널 3: 위험 키워드 탐지 ──
        detected, keyword_score = self._detect_keywords(utterances)

        # ── 시그널 4: 부정 감정 연속성 ──
        streak_score = self._calc_negative_streak(groups)

        # ── 시그널 5: 대화 종료 감정 ──
        ending_score = self._calc_ending_score(groups)

        # ── 최종 점수 산출 ──
        signals = {
            "negative_ratio": neg_ratio,
            "anger_ratio": anger_ratio,
            "keyword_score": keyword_score,
            "streak_score": streak_score,
            "ending_score": ending_score,
        }

        raw_score = sum(
            signals[key] * SIGNAL_WEIGHTS[key] for key in SIGNAL_WEIGHTS
        )
        final_score = round(min(max(raw_score, 0.0), 1.0), 4)

        # 등급 결정
        level_info = self._score_to_level(final_score)

        return RiskResult(
            dialogue_id=dialogue_id or emotion_result.dialogue_id,
            risk_score=final_score,
            risk_level=level_info["label_en"],
            risk_label=level_info["label"],
            risk_grade=level_info["level"],
            signals=signals,
            detected_keywords=detected,
            emotion_sequence=emotion_seq,
            recommendation=RISK_RECOMMENDATIONS.get(level_info["label_en"], ""),
            method="rule_based",
        )

    def _detect_keywords(
        self, utterances: list[str]
    ) -> tuple[list[DetectedKeyword], float]:
        """
        발화 리스트에서 위험 키워드를 탐지하고 점수를 산출한다.

        Returns
        -------
        tuple[list[DetectedKeyword], float]
            (탐지된 키워드 리스트, 위험도 점수 0~1)
        """
        detected: list[DetectedKeyword] = []
        max_weight = 0.0

        for idx, utt in enumerate(utterances):
            for kp in self._keyword_patterns:
                if kp["pattern"].search(utt):
                    detected.append(DetectedKeyword(
                        keyword=kp["keyword"],
                        severity=kp["severity"],
                        weight=kp["weight"],
                        utterance_index=idx,
                    ))
                    max_weight = max(max_weight, kp["weight"])

        if not detected:
            return detected, 0.0

        # 점수 산출: 최고 위험 등급 키워드 가중치를 기본으로,
        # 탐지 건수에 따라 보정 (최대 1.0)
        count_factor = min(len(detected) / 5.0, 1.0)  # 5건 이상이면 1.0
        score = round(max_weight * 0.7 + count_factor * 0.3, 4)
        return detected, min(score, 1.0)

    @staticmethod
    def _calc_negative_streak(groups: list[str]) -> float:
        """
        부정 감정 연속 최대 길이 기반 점수 산출.

        Returns
        -------
        float
            0~1 범위의 연속성 점수.
        """
        if not groups:
            return 0.0

        max_streak = 0
        current = 0
        for g in groups:
            if g == "negative":
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        # 3회 연속이면 0.6, 5회 이상이면 1.0
        if max_streak >= 5:
            return 1.0
        elif max_streak >= 3:
            return 0.6 + (max_streak - 3) * 0.2
        elif max_streak >= 2:
            return 0.4
        elif max_streak >= 1:
            return 0.2
        return 0.0

    @staticmethod
    def _calc_ending_score(groups: list[str]) -> float:
        """
        대화 종료 시점 감정 기반 점수 산출.

        마지막 1~2 발화의 감정을 기준으로 판단.

        Returns
        -------
        float
            0~1 범위의 종료 감정 점수.
        """
        if not groups:
            return 0.0

        last = groups[-1]
        second_last = groups[-2] if len(groups) >= 2 else None

        # 마지막이 부정이면 기본 0.8
        if last == "negative":
            score = 0.8
            # 마지막 2개 모두 부정이면 1.0
            if second_last == "negative":
                score = 1.0
            return score
        elif last == "neutral":
            # 중립이지만 직전이 부정이면 0.3
            if second_last == "negative":
                return 0.3
            return 0.1
        else:
            # 긍정으로 끝남
            return 0.0

    @staticmethod
    def _score_to_level(score: float) -> dict:
        """점수를 위험도 등급으로 변환한다."""
        for level in RISK_LEVELS:
            if score < level["max"] or (score == 1.0 and level["level"] == 5):
                return level
        return RISK_LEVELS[-1]


# ──────────────────────────────────────────────
# 6. LLM 기반 위험도 분석기
# ──────────────────────────────────────────────

class LLMRiskAnalyzer:
    """
    LLM 프롬프트 기반 위험도 분석기.

    특징:
      - Chain-of-Thought 추론으로 다각도 위험 요소 분석
      - 문맥·뉘앙스·비유를 고려한 정밀 판단
      - Rule-Based 결과를 참고 정보로 제공하여 정확도 향상
    """

    RISK_ANALYSIS_PROMPT = """당신은 **연인 갈등 상황** 전문 위험도 분석가입니다.

## 과업
아래 대화의 **갈등 위험도**를 분석하세요.

## 위험도 5단계
| 등급 | 라벨 | 점수 범위 | 설명 |
|---|---|---|---|
| 1 | 안전(safe) | 0.0~0.2 | 갈등 징후 없음. 일상 대화 |
| 2 | 주의(caution) | 0.2~0.4 | 경미한 불만/서운함 |
| 3 | 경고(warning) | 0.4~0.6 | 명확한 갈등 신호. 분노/혐오 반복 |
| 4 | 위험(danger) | 0.6~0.8 | 강한 갈등. 공격적 표현, 감정 폭발 |
| 5 | 심각(critical) | 0.8~1.0 | 극단적 갈등. 관계 단절 위험 |

## 분석 관점 (Chain-of-Thought)
다음 관점을 모두 고려하여 단계별로 추론하세요:

1. **감정 강도**: 부정 감정(분노, 혐오, 슬픔, 공포)의 빈도와 강도
2. **표현 수위**: 공격적·폭력적·극단적 표현의 유무와 정도
3. **갈등 구조**: 일방적 공격인지, 상호 갈등인지, 오해 기반인지
4. **관계 위협**: 관계 단절·이별 암시 표현의 유무
5. **감정 흐름**: 감정이 악화되고 있는지, 진정되고 있는지
6. **대화 종료 방향**: 마지막 발화의 감정이 부정적인지 긍정적인지

## Rule-Based 참고 결과
{rule_based_info}

## 대화 내용
{dialogue}

## 감정 시퀀스 (참고)
{emotion_sequence}

## 출력 형식 (반드시 아래 JSON만 출력)
```json
{{
  "risk_score": 0.0~1.0,
  "risk_level": "safe|caution|warning|danger|critical",
  "risk_label": "한글 라벨",
  "risk_grade": 1~5,
  "analysis": {{
    "emotion_intensity": "감정 강도 분석 (1문장)",
    "expression_level": "표현 수위 분석 (1문장)",
    "conflict_structure": "갈등 구조 분석 (1문장)",
    "relationship_threat": "관계 위협 수준 (1문장)",
    "emotion_trend": "감정 흐름 분석 (1문장)",
    "ending_direction": "종료 방향 분석 (1문장)"
  }},
  "recommendation": "대응 전략 (1~2문장)",
  "reasoning": "종합 판단 근거 (2~3문장)"
}}
```"""

    def get_prompt(
        self,
        utterances: list[str],
        emotion_sequence: list[str],
        rule_based_result: RiskResult | None = None,
    ) -> str:
        """위험도 분석 프롬프트를 생성한다."""
        dialogue_text = "\n".join(
            f"[발화 {i}] {u}" for i, u in enumerate(utterances)
        )
        seq_text = " > ".join(emotion_sequence) if emotion_sequence else "(미분석)"

        if rule_based_result:
            rule_info = (
                f"- 점수: {rule_based_result.risk_score}\n"
                f"- 등급: {rule_based_result.risk_label} ({rule_based_result.risk_level})\n"
                f"- 시그널: {json.dumps(rule_based_result.signals, ensure_ascii=False)}\n"
                f"- 탐지 키워드: {', '.join(dk.keyword for dk in rule_based_result.detected_keywords) or '없음'}"
            )
        else:
            rule_info = "(Rule-Based 결과 없음)"

        return self.RISK_ANALYSIS_PROMPT.format(
            dialogue=dialogue_text,
            emotion_sequence=seq_text,
            rule_based_info=rule_info,
        )

    def parse_response(
        self, llm_output: str, dialogue_id: str | None = None,
        emotion_sequence: list[str] | None = None,
    ) -> RiskResult:
        """
        LLM 응답을 파싱하여 RiskResult로 변환한다.

        Parameters
        ----------
        llm_output : str
            LLM의 JSON 응답 문자열.
        dialogue_id : str | None
            대화 식별자.
        emotion_sequence : list[str] | None
            감정 시퀀스.

        Returns
        -------
        RiskResult
            파싱된 위험도 분석 결과.
        """
        parsed = self._extract_json(llm_output)

        risk_score = float(parsed.get("risk_score", 0.0))
        risk_level = parsed.get("risk_level", "safe")
        risk_label = parsed.get("risk_label", "안전")
        risk_grade = int(parsed.get("risk_grade", 1))
        analysis = parsed.get("analysis", {})
        recommendation = parsed.get("recommendation", "")
        reasoning = parsed.get("reasoning", "")

        # 분석 내용을 reasoning에 통합
        if analysis:
            analysis_text = " / ".join(
                f"{k}: {v}" for k, v in analysis.items() if v
            )
            reasoning = f"{reasoning} [상세] {analysis_text}"

        return RiskResult(
            dialogue_id=dialogue_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            risk_label=risk_label,
            risk_grade=risk_grade,
            signals={},  # LLM은 별도 시그널 없음
            detected_keywords=[],
            emotion_sequence=emotion_sequence or [],
            recommendation=recommendation,
            method="llm",
            reasoning=reasoning,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """LLM 출력에서 JSON 블록을 추출하여 파싱한다."""
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
# 7. 통합 분석 함수
# ──────────────────────────────────────────────

def analyze_risk(
    utterances: list[str],
    dialogue_id: str | None = None,
    emotion_result: DialogueEmotionResult | None = None,
    method: str = "rule_based",
    llm_caller=None,
) -> RiskResult:
    """
    대화 위험도 분석 통합 함수.

    Parameters
    ----------
    utterances : list[str]
        대화 발화 리스트.
    dialogue_id : str | None
        대화 식별자.
    emotion_result : DialogueEmotionResult | None
        사전 수행된 감정 분석 결과. None이면 내부에서 수행.
    method : str
        분석 방식. "rule_based" 또는 "llm".
    llm_caller : callable | None
        LLM 호출 함수. method="llm" 시 필수.

    Returns
    -------
    RiskResult
        구조화된 위험도 분석 결과.
    """
    if method == "rule_based":
        analyzer = RuleBasedRiskAnalyzer()
        return analyzer.analyze(utterances, emotion_result, dialogue_id)

    elif method == "llm":
        if llm_caller is None:
            raise ValueError("LLM 방식 사용 시 llm_caller 함수를 제공해야 합니다.")

        # Rule-Based 결과를 먼저 산출하여 LLM에 참고 정보로 제공
        rb_analyzer = RuleBasedRiskAnalyzer()
        rb_result = rb_analyzer.analyze(utterances, emotion_result, dialogue_id)

        llm_analyzer = LLMRiskAnalyzer()
        prompt = llm_analyzer.get_prompt(
            utterances=utterances,
            emotion_sequence=rb_result.emotion_sequence,
            rule_based_result=rb_result,
        )
        response = llm_caller(prompt)
        return llm_analyzer.parse_response(
            response,
            dialogue_id=dialogue_id,
            emotion_sequence=rb_result.emotion_sequence,
        )

    else:
        raise ValueError(f"지원하지 않는 분석 방식: {method}")


def cross_validate_risk(
    utterances: list[str],
    dialogue_id: str | None = None,
    emotion_result: DialogueEmotionResult | None = None,
    llm_caller=None,
) -> dict:
    """
    Rule-Based와 LLM 위험도 결과를 교차 검증한다.

    Parameters
    ----------
    utterances : list[str]
        대화 발화 리스트.
    dialogue_id : str | None
        대화 식별자.
    emotion_result : DialogueEmotionResult | None
        사전 수행된 감정 분석 결과.
    llm_caller : callable
        LLM 호출 함수.

    Returns
    -------
    dict
        교차 검증 결과 (rule_based, llm, match, final, note).
    """
    rule_result = analyze_risk(
        utterances, dialogue_id, emotion_result, method="rule_based"
    )

    if llm_caller is None:
        return {
            "rule_based": rule_result.to_dict(),
            "llm": None,
            "match": None,
            "final": rule_result.to_dict(),
            "conflict_note": "LLM 미사용 — Rule-Based 결과 단독 채택",
        }

    llm_result = analyze_risk(
        utterances, dialogue_id, emotion_result,
        method="llm", llm_caller=llm_caller,
    )

    # 등급 차이로 일치 여부 판단 (±1 등급 이내면 일치로 간주)
    grade_diff = abs(rule_result.risk_grade - llm_result.risk_grade)
    is_match = grade_diff <= 1

    if is_match:
        # 평균 점수로 최종 결정
        avg_score = round((rule_result.risk_score + llm_result.risk_score) / 2, 4)
        final = rule_result
        final.risk_score = avg_score
        level_info = RuleBasedRiskAnalyzer._score_to_level(avg_score)
        final.risk_level = level_info["label_en"]
        final.risk_label = level_info["label"]
        final.risk_grade = level_info["level"]
        note = (
            f"Rule-Based({rule_result.risk_grade}등급)와 LLM({llm_result.risk_grade}등급) "
            f"결과 일치(차이 ≤ 1) → 평균 점수({avg_score}) 채택"
        )
    else:
        # 불일치 시 더 높은(보수적인) 등급 채택
        if llm_result.risk_grade > rule_result.risk_grade:
            final = llm_result
        else:
            final = rule_result
        note = (
            f"불일치 발생 (Rule: {rule_result.risk_grade}등급, LLM: {llm_result.risk_grade}등급). "
            f"보수적 판단 원칙에 따라 상위 등급({final.risk_grade}등급) 채택"
        )

    return {
        "rule_based": rule_result.to_dict(),
        "llm": llm_result.to_dict(),
        "match": is_match,
        "final": final.to_dict(),
        "conflict_note": note,
    }


# ──────────────────────────────────────────────
# 8. 통합 파이프라인: 감정 + 위험도
# ──────────────────────────────────────────────

def full_analysis(
    utterances: list[str],
    dialogue_id: str | None = None,
    method: str = "rule_based",
    llm_caller=None,
) -> dict:
    """
    감정 분석 → 위험도 분석을 연결하는 통합 파이프라인.

    Parameters
    ----------
    utterances : list[str]
        대화 발화 리스트.
    dialogue_id : str | None
        대화 식별자.
    method : str
        분석 방식. "rule_based" 또는 "llm".
    llm_caller : callable | None
        LLM 호출 함수.

    Returns
    -------
    dict
        emotion: 감정 분석 결과, risk: 위험도 분석 결과.
    """
    from .emotion_analyzer import analyze_dialogue_emotion

    # 1단계: 감정 분석
    emotion_result = analyze_dialogue_emotion(
        utterances, dialogue_id, method=method, llm_caller=llm_caller
    )

    # 2단계: 위험도 분석
    risk_result = analyze_risk(
        utterances, dialogue_id, emotion_result,
        method=method, llm_caller=llm_caller
    )

    return {
        "dialogue_id": dialogue_id,
        "emotion": emotion_result.to_dict(),
        "risk": risk_result.to_dict(),
    }


# ──────────────────────────────────────────────
# 9. CLI 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("위험도 분석기 테스트 (Rule-Based)")
    print("=" * 60)

    # 테스트 케이스 1: 안전한 대화
    safe_dialogue = [
        "오늘 뭐 먹을까?",
        "글쎄, 뭐 먹고 싶어?",
        "파스타 어때?",
        "좋지! 맛있겠다.",
    ]

    # 테스트 케이스 2: 주의 수준 대화
    caution_dialogue = [
        "오늘도 늦었네...",
        "미안. 회사가 바빠서.",
        "맨날 바쁘다 바쁘다... 답답해.",
        "알았어. 다음부턴 미리 얘기할게.",
    ]

    # 테스트 케이스 3: 경고 수준 대화
    warning_dialogue = [
        "왜 여자 번호가 이렇게 많아?",
        "그냥 회사 사람들이야. 왜 그래?",
        "왜 그래? 니가 수상하니까 그렇지!",
        "의심하지 마. 짜증나네.",
        "짜증나는 건 나라고!",
    ]

    # 테스트 케이스 4: 위험 수준 대화
    danger_dialogue = [
        "나 자기 마누라 찾아가서 담판 질거야",
        "지금, 누굴 찾아가 담판을 져",
        "나랑 그동안의 관계 다 말하고. 이혼해 달라고 할거야",
        "이 시대에 뽀뽀 몇 번 한 것도 책임져야 돼냐?",
        "아무튼 난 절대 자기 포기 못해, 절대!!!",
    ]

    # 테스트 케이스 5: 심각 수준 대화
    critical_dialogue = [
        "이제 진짜 끝내자. 지긋지긋해.",
        "뭐? 갑자기 왜 이래.",
        "갑자기가 아니야. 한참 참은 거야. 꺼져.",
        "나보고 꺼지라고?",
        "보기 싫어. 헤어지자. 죽겠다 진짜.",
    ]

    test_cases = [
        ("안전", safe_dialogue),
        ("주의", caution_dialogue),
        ("경고", warning_dialogue),
        ("위험", danger_dialogue),
        ("심각", critical_dialogue),
    ]

    for label, dialogue in test_cases:
        result = full_analysis(dialogue, dialogue_id=f"test_{label}")

        emotion = result["emotion"]
        risk = result["risk"]

        print(f"\n{'─' * 50}")
        print(f"[기대 등급: {label}]")
        print(f"  감정 시퀀스: {' > '.join(emotion['emotion_sequence'])}")
        print(f"  지배적 감정: {emotion['dominant_emotion']} ({emotion['dominant_group']})")
        print(f"  부정 비율: {emotion['negative_ratio']}")
        print(f"  위험도 점수: {risk['risk_score']}")
        print(f"  위험도 등급: {risk['risk_label']} ({risk['risk_level']}, {risk['risk_grade']}등급)")
        print(f"  대응 전략: {risk['recommendation']}")

        if risk['detected_keywords']:
            kw_list = [
                f"{dk['keyword']}({dk['severity']})"
                for dk in risk['detected_keywords']
            ]
            print(f"  탐지 키워드: {', '.join(kw_list)}")
        else:
            print(f"  탐지 키워드: 없음")

    # LLM 프롬프트 예시
    print("\n" + "=" * 60)
    print("LLM 위험도 분석 프롬프트 예시")
    print("=" * 60)
    llm_analyzer = LLMRiskAnalyzer()
    rb_analyzer = RuleBasedRiskAnalyzer()
    rb_result = rb_analyzer.analyze(critical_dialogue)
    prompt = llm_analyzer.get_prompt(
        utterances=critical_dialogue,
        emotion_sequence=rb_result.emotion_sequence,
        rule_based_result=rb_result,
    )
    print(prompt[:800] + "...")
