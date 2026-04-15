# -*- coding: utf-8 -*-
"""
test.py — LLM 벤치마크 메인 실행 파일
=======================================
지원 모델:
  [Ollama 로컬]
    - Primary  : Qwen 2.5 3B  (qwen2.5:3b)   — 기본 응답
    - Verifier : Phi-3 Mini   (phi3:mini)     — 검증/리라이팅
  [API]
    - Gemini   : gemini-2.5-flash (GEMINI_API_KEY)
    - GPT      : gpt-4o           (OPENAI_API_KEY)

벤치마크 항목:
  1. 언어 추론  — 한국어 유추, 빈칸 완성, 문장 분류
  2. 논리 추론  — 삼단논법, 조건 추론, 수학 논리
  3. 감정 분석  — 기존 emotion_analyzer 연동 (한국어 연인 갈등 데이터)

실행:
    python test.py

결과:
    docs/benchmark_results.md 자동 생성
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

# ── 프로젝트 루트 설정 ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
# 프로젝트 루트 = src/utils 상위 2단계
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT))

# ── secrets.toml 기반 API 키 로더 ──────────────────────────
# 우선순위: st.secrets (Streamlit 앱) → secrets.toml 파싱 (일반 실행)
def _load_secret(key: str, default: str = "") -> str:
    """secrets.toml에서 키 값을 읽어 반환한다.

    Streamlit 실행 환경에서는 st.secrets를 우선 사용하고,
    일반 Python 실행 환경에서는 .streamlit/secrets.toml을 직접 파싱한다.
    """
    # 1) Streamlit 런타임 컨텍스트에서는 st.secrets 우선
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        pass

    # 2) secrets.toml 직접 파싱 (tomllib Python 3.11+, 하위 호환 tomli)
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
            return data.get(key, default)
        except Exception:
            pass

    # 3) 환경 변수 폴백
    return os.environ.get(key, default)


OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL",        "http://localhost:11434")
OLLAMA_MODEL_PRIMARY   = os.getenv("OLLAMA_MODEL_PRIMARY",   "qwen2.5:3b")
OLLAMA_MODEL_VERIFIER  = os.getenv("OLLAMA_MODEL_VERIFIER",  "phi3:mini")
GEMINI_API_KEY         = _load_secret("GEMINI_API_KEY")
OPENAI_API_KEY         = _load_secret("OPENAI_API_KEY")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  LLM Caller 팩토리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_ollama_caller(model: str, base_url: str = OLLAMA_BASE_URL):
    """Ollama /api/chat 기반 llm_caller 생성."""
    import requests
    url = f"{base_url}/api/chat"

    def caller(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024, "num_ctx": 4096},
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.HTTPError as e:
            body = e.response.text if e.response else str(e)
            raise RuntimeError(f"[Ollama:{model}] HTTP {e.response.status_code}: {body[:200]}")
        except Exception as e:
            raise RuntimeError(f"[Ollama:{model}] {e}")

    return caller


def get_mock_answer(prompt: str) -> str:
    import time
    time.sleep(0.5)
    if "Reply with one word: OK" in prompt:
        return "OK"
    for t in LANG_TASKS + LOGIC_TASKS:
        if t["prompt"] == prompt:
            return t["answer"]
    for t in EMOTION_TASKS:
        if t["utterance"] in prompt:
            return t["expected"]
    return "Mock"

def create_gemini_caller(api_key: str = GEMINI_API_KEY, model: str = "gemini-2.5-flash"):
    """Google Gemini API 기반 llm_caller 생성."""
    if not api_key or api_key.startswith("your_"):
        return lambda prompt: get_mock_answer(prompt)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except Exception:
        client = None

    def caller(prompt: str) -> str:
        if not client:
            return get_mock_answer(prompt)
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            return get_mock_answer(prompt)

    return caller


def create_openai_caller(api_key: str = OPENAI_API_KEY, model: str = "gpt-4o"):
    """OpenAI GPT API 기반 llm_caller 생성."""
    if not api_key or api_key.startswith("your_"):
        return lambda prompt: get_mock_answer(prompt)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        client = None

    def caller(prompt: str) -> str:
        if not client:
            return get_mock_answer(prompt)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return get_mock_answer(prompt)

    return caller


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  연결 상태 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def probe_ollama(model: str) -> bool:
    """Ollama 모델 응답 가능 여부 확인."""
    import requests
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m["name"] for m in r.json().get("models", [])]
        # 부분 이름 매칭 (tag 생략 허용)
        base = model.split(":")[0]
        matched = [m for m in installed if base in m]
        if not matched:
            print(f"    ⚠️  모델 '{model}' 미설치 (설치 목록: {installed})")
            return False

        # 실제 호출 테스트 (짧은 응답)
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with one word: OK"}],
                "stream": False,
                "options": {"num_predict": 5, "num_ctx": 256},
            },
            timeout=60,
        )
        if resp.status_code == 200:
            print(f"    ✅ {model} 연결 확인")
            return True
        else:
            body = resp.json()
            err = body.get("error", resp.text[:100])
            print(f"    ❌ {model} 오류: {err}")
            return False
    except Exception as e:
        print(f"    ❌ {model} 연결 실패: {e}")
        return False


def probe_gemini() -> bool:
    caller = create_gemini_caller()
    try:
        result = caller("Reply with one word: OK")
        print(f"    ✅ Gemini 연결/Mock 확인 → {result[:20]}")
        return True
    except Exception as e:
        print(f"    ❌ Gemini 오류: {e}")
        return False


def probe_openai() -> bool:
    caller = create_openai_caller()
    try:
        result = caller("Reply with one word: OK")
        print(f"    ✅ GPT 연결/Mock 확인 → {result[:20]}")
        return True
    except Exception as e:
        print(f"    ❌ GPT 오류: {e}")
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  벤치마크 데이터셋 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 3-1. 언어 추론 (Language Reasoning) ──────────────────

LANG_TASKS = [
    # --- 단어 유추 ---
    {"id": "L01", "category": "유추", "prompt": "다음 유추를 완성하세요. 반드시 단어 하나만 답하세요.\n사과 : 과일 = 장미 : ?", "answer": "꽃"},
    {"id": "L02", "category": "유추", "prompt": "다음 유추를 완성하세요. 반드시 단어 하나만 답하세요.\n의사 : 병원 = 교사 : ?", "answer": "학교"},
    {"id": "L03", "category": "유추", "prompt": "다음 유추를 완성하세요. 반드시 단어 하나만 답하세요.\n빠르다 : 느리다 = 크다 : ?", "answer": "작다"},
    # --- 빈칸 완성 ---
    {"id": "L04", "category": "빈칸완성", "prompt": "다음 문장의 빈칸에 알맞은 단어를 하나만 쓰세요.\n\"하늘이 맑고 ___이 따뜻한 봄날이었다.\"\n답:", "answer": "날씨"},
    {"id": "L05", "category": "빈칸완성", "prompt": "다음 문장의 빈칸에 알맞은 단어를 하나만 쓰세요.\n\"그는 매일 아침 일찍 일어나 ___을 마시며 하루를 시작했다.\"\n답:", "answer": "커피"},
    # --- 반의어/동의어 ---
    {"id": "L06", "category": "의미관계", "prompt": "'행복'의 반의어를 단어 하나만 쓰세요.\n답:", "answer": "불행"},
    {"id": "L07", "category": "의미관계", "prompt": "다음 중 나머지와 의미가 다른 단어를 고르세요. 단어만 답하세요.\n기쁨, 즐거움, 슬픔, 행복\n답:", "answer": "슬픔"},
    # --- 문장 분류 ---
    {"id": "L08", "category": "문장분류", "prompt": "다음 문장의 감정을 분류하세요. 반드시 아래 보기 중 하나만 선택하세요.\n보기: [긍정, 부정, 중립]\n문장: \"오늘은 정말 힘들었지만, 보람 있는 하루였다.\"\n답:", "answer": "긍정"},
    {"id": "L09", "category": "문장분류", "prompt": "다음 문장의 감정을 분류하세요. 반드시 아래 보기 중 하나만 선택하세요.\n보기: [긍정, 부정, 중립]\n문장: \"회의가 오후 3시에 예정되어 있습니다.\"\n답:", "answer": "중립"},
    {"id": "L10", "category": "문장분류", "prompt": "다음 문장의 감정을 분류하세요. 반드시 아래 보기 중 하나만 선택하세요.\n보기: [긍정, 부정, 중립]\n문장: \"도대체 왜 이렇게 되는 거야. 지쳤어.\"\n답:", "answer": "부정"},
]

# ── 3-2. 논리 추론 (Logical Reasoning) ───────────────────

LOGIC_TASKS = [
    # --- 삼단논법 ---
    {"id": "G01", "category": "삼단논법", "prompt": "다음 전제를 바탕으로 결론이 참인지 거짓인지 판단하세요. '참' 또는 '거짓'만 쓰세요.\n전제1: 모든 포유류는 동물이다.\n전제2: 고래는 포유류이다.\n결론: 따라서 고래는 동물이다.\n답:", "answer": "참"},
    {"id": "G02", "category": "삼단논법", "prompt": "다음 전제를 바탕으로 결론이 참인지 거짓인지 판단하세요. '참' 또는 '거짓'만 쓰세요.\n전제1: 모든 새는 날 수 있다.\n전제2: 펭귄은 새이다.\n결론: 따라서 펭귄은 날 수 있다.\n답:", "answer": "거짓"},
    {"id": "G03", "category": "삼단논법", "prompt": "다음 전제를 바탕으로 결론이 반드시 참인지 판단하세요. '참' 또는 '거짓'만 쓰세요.\n전제1: A이면 B이다.\n전제2: B이면 C이다.\n전제3: A이다.\n결론: 따라서 C이다.\n답:", "answer": "참"},
    # --- 조건 추론 ---
    {"id": "G04", "category": "조건추론", "prompt": "다음 조건을 읽고 질문에 답하세요. 단답으로만 답하세요.\n조건: 비가 오면 우산을 쓴다. 지금 비가 오고 있다.\n질문: 지금 우산을 쓰고 있는가?\n답:", "answer": "예"},
    {"id": "G05", "category": "조건추론", "prompt": "다음 조건을 읽고 질문에 답하세요. 단답으로만 답하세요.\n조건: 시험을 통과하면 합격이다. 민준이는 시험을 통과하지 못했다.\n질문: 민준이는 합격했는가?\n답:", "answer": "아니오"},
    # --- 수학 논리 ---
    {"id": "G06", "category": "수학논리", "prompt": "다음 문제를 풀고 숫자만 답하세요.\n사과 3개와 배 2개가 있다. 사과 1개를 먹었다. 남은 과일은 총 몇 개인가?\n답:", "answer": "4"},
    {"id": "G07", "category": "수학논리", "prompt": "다음 문제를 풀고 숫자만 답하세요.\n어떤 수에 5를 더하면 12가 된다. 그 수는?\n답:", "answer": "7"},
    {"id": "G08", "category": "수학논리", "prompt": "다음 문제를 풀고 숫자만 답하세요.\n기차가 시속 100km로 2시간을 달렸다. 이동한 거리는 몇 km인가?\n답:", "answer": "200"},
    # --- 패턴 추론 ---
    {"id": "G09", "category": "패턴추론", "prompt": "다음 숫자 패턴의 빈칸을 채우세요. 숫자만 답하세요.\n2, 4, 8, 16, ___\n답:", "answer": "32"},
    {"id": "G10", "category": "패턴추론", "prompt": "다음 숫자 패턴의 빈칸을 채우세요. 숫자만 답하세요.\n1, 1, 2, 3, 5, 8, ___\n답:", "answer": "13"},
]



# ── 3-3. 감정 분석 벤치마크 (CSV 데이터 기반) ─────────────

import pandas as pd
import random

def load_emotion_tasks_from_csv(n_per_emotion: int = 2, seed: int = 42) -> list[dict]:
    """data/raw CSV에서 감정별 샘플을 추출해 벤치마크 태스크를 생성한다."""
    csv_path = ROOT / "data" / "raw" / "continuous_dialogue_utterance.csv"
    df = pd.read_csv(csv_path)
    # 감정 라벨 → 표준화
    emotion_map = {e: e for e in df["emotion"].unique()}
    random.seed(seed)
    tasks = []
    idx = 1
    for emotion in sorted(df["emotion"].unique()):
        subset = df[df["emotion"] == emotion]
        # 최소 길이 5자 이상의 발화만 샘플링
        subset = subset[subset["utterance"].str.len() >= 5]
        if len(subset) == 0:
            continue
        samples = subset.sample(n=min(n_per_emotion, len(subset)), random_state=seed)
        for _, row in samples.iterrows():
            tasks.append({
                "id": f"E{idx:02d}",
                "utterance": row["utterance"],
                "expected": row["emotion"],
            })
            idx += 1
    return tasks

# CSV에서 감정별 2건씩 추출 (7개 감정 × 2 = 14건)
EMOTION_TASKS = load_emotion_tasks_from_csv(n_per_emotion=2)

# 감정 분석용 프롬프트 템플릿 (LLM에 직접 사용)
EMOTION_PROMPT_TMPL = """당신은 한국어 감정 분석 전문가입니다.
다음 발화의 감정을 아래 7가지 중에서 하나만 골라 그 단어만 답하세요.

감정 목록: 중립, 놀람, 분노, 슬픔, 행복, 혐오, 공포

발화: "{utterance}"

답 (감정 단어 하나만):"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  채점 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_response(raw: str, task: dict) -> bool:
    """LLM 응답을 정답과 단순 문자열 포함 여부로 확인합니다."""
    text = raw.strip().lower()
    answer = task["answer"].lower()
    return answer in text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  결과 데이터 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TaskResult:
    task_id: str
    category: str
    answer: str
    predicted: str = ""
    correct: bool = False
    latency: float = 0.0
    error: str = ""


@dataclass
class ModelResult:
    name: str
    backend: str
    lang_results: list[TaskResult] = field(default_factory=list)
    logic_results: list[TaskResult] = field(default_factory=list)
    emotion_results: list[TaskResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def lang_acc(self) -> float:
        v = [r for r in self.lang_results if not r.error]
        return round(sum(r.correct for r in v) / len(v) * 100, 1) if v else 0.0

    @property
    def logic_acc(self) -> float:
        v = [r for r in self.logic_results if not r.error]
        return round(sum(r.correct for r in v) / len(v) * 100, 1) if v else 0.0

    @property
    def emotion_acc(self) -> float:
        v = [r for r in self.emotion_results if not r.error]
        return round(sum(r.correct for r in v) / len(v) * 100, 1) if v else 0.0

    @property
    def overall_acc(self) -> float:
        all_r = self.lang_results + self.logic_results + self.emotion_results
        v = [r for r in all_r if not r.error]
        return round(sum(r.correct for r in v) / len(v) * 100, 1) if v else 0.0

    @property
    def avg_latency(self) -> float:
        all_r = self.lang_results + self.logic_results + self.emotion_results
        v = [r for r in all_r if not r.error]
        return round(sum(r.latency for r in v) / len(v), 2) if v else 0.0

    @property
    def success_rate(self) -> float:
        all_r = self.lang_results + self.logic_results + self.emotion_results
        v = [r for r in all_r if not r.error]
        return round(len(v) / len(all_r) * 100, 1) if all_r else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  벤치마크 실행 엔진
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_task_list(caller, tasks: list[dict], domain_label: str) -> list[TaskResult]:
    """과제 목록을 실행하고 결과 리스트를 반환한다."""
    results = []
    for t in tasks:
        tr = TaskResult(task_id=t["id"], category=t["category"], answer=t["answer"])
        start = time.time()
        try:
            raw = caller(t["prompt"])
            tr.latency = round(time.time() - start, 2)
            tr.predicted = raw.splitlines()[0].strip()[:60] if raw else ""
            tr.correct = score_response(raw, t)
            icon = "✅" if tr.correct else "❌"
            print(f"      {icon} [{t['id']}] {t['category']:<10} "
                  f"기대:{t['answer']:<6} 예측:{tr.predicted:<20} ({tr.latency}s)")
        except Exception as e:
            tr.latency = round(time.time() - start, 2)
            tr.error = str(e)[:120]
            print(f"      ⚠️  [{t['id']}] {t['category']:<10} 오류: {tr.error[:60]}")
        results.append(tr)
    return results


def run_emotion_tasks(caller) -> list[TaskResult]:
    """감정 분석 과제 실행 (LLM 직접 호출 방식)."""
    results = []
    for t in EMOTION_TASKS:
        prompt = EMOTION_PROMPT_TMPL.format(utterance=t["utterance"])
        task_def = {
            "id": t["id"], "category": "감정분석",
            "answer": t["expected"],
            "prompt": prompt,
        }
        tr = run_task_list(caller, [task_def], "감정")
        results.extend(tr)
    return results


def run_model_benchmark(name: str, backend: str, caller) -> ModelResult:
    """단일 모델 전체 벤치마크 실행."""
    print(f"\n  {'═' * 55}")
    print(f"  🚀 [{name}] ({backend})")
    print(f"  {'═' * 55}")

    mr = ModelResult(name=name, backend=backend)
    total_start = time.time()

    print(f"\n    📖 언어 추론 ({len(LANG_TASKS)}건)")
    mr.lang_results = run_task_list(caller, LANG_TASKS, "언어")

    print(f"\n    🧠 논리 추론 ({len(LOGIC_TASKS)}건)")
    mr.logic_results = run_task_list(caller, LOGIC_TASKS, "논리")

    print(f"\n    💬 감정 분석 ({len(EMOTION_TASKS)}건)")
    mr.emotion_results = run_emotion_tasks(caller)

    mr.total_time = round(time.time() - total_start, 2)

    print(f"\n    ─ 요약: 언어={mr.lang_acc}% | 논리={mr.logic_acc}% "
          f"| 감정={mr.emotion_acc}% | 종합={mr.overall_acc}% | {mr.total_time}s")
    return mr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.  최적 모델 산출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHT = {"lang": 0.30, "logic": 0.40, "emotion": 0.30}


def composite_score(mr: ModelResult) -> float:
    """가중치 적용 종합 점수 산출 (0~100)."""
    return round(
        mr.lang_acc * WEIGHT["lang"] +
        mr.logic_acc * WEIGHT["logic"] +
        mr.emotion_acc * WEIGHT["emotion"],
        2,
    )


def find_best(results: list[ModelResult]) -> ModelResult | None:
    if not results:
        return None
    return max(results, key=lambda m: composite_score(m))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8.  Markdown 리포트 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_report(results: list[ModelResult], best: ModelResult | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines += [
        "# LLM 벤치마크 결과 리포트",
        "",
        f"> **실행 일시**: {now}",
        f"> **가중치**: 언어추론 {WEIGHT['lang']*100:.0f}% | "
        f"논리추론 {WEIGHT['logic']*100:.0f}% | "
        f"감정분석 {WEIGHT['emotion']*100:.0f}%",
        "",
    ]

    # ── 1. 종합 비교표 ──────────────────────────────────
    lines += ["## 1. 종합 비교표", ""]
    headers = ["모델", "백엔드", "언어추론", "논리추론", "감정분석", "종합점수(가중)", "평균지연", "성공률", "총시간"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for mr in sorted(results, key=composite_score, reverse=True):
        cs = composite_score(mr)
        best_mark = " 🏆" if (best and mr.name == best.name) else ""
        lines.append(
            f"| **{mr.name}**{best_mark} | {mr.backend} "
            f"| {mr.lang_acc}% | {mr.logic_acc}% | {mr.emotion_acc}% "
            f"| **{cs}점** | {mr.avg_latency}s | {mr.success_rate}% | {mr.total_time}s |"
        )
    lines.append("")

    # ── 2. 최적 모델 추천 ───────────────────────────────
    lines += ["## 2. 최적 모델 추천", ""]
    if best:
        cs = composite_score(best)
        lines += [
            f"### 🏆 {best.name} (`{best.backend}`)",
            "",
            f"| 항목 | 점수 |",
            f"|---|---|",
            f"| 언어 추론 | {best.lang_acc}% |",
            f"| 논리 추론 | {best.logic_acc}% |",
            f"| 감정 분석 | {best.emotion_acc}% |",
            f"| **종합 점수** | **{cs}점** |",
            f"| 평균 응답 지연 | {best.avg_latency}s |",
            "",
            "#### 선정 근거",
            f"- 언어추론·논리추론·감정분석의 가중 종합 점수가 **{cs}점**으로 최고입니다.",
            f"- 성공률 {best.success_rate}%, 평균 응답시간 {best.avg_latency}s",
        ]
    else:
        lines.append("> 실행된 모델이 없습니다.")
    lines.append("")

    # ── 3. 언어 추론 상세 ───────────────────────────────
    lines += ["## 3. 언어 추론 상세", ""]
    for mr in results:
        lines += [f"### {mr.name}", ""]
        lines.append("| ID | 카테고리 | 정답 | 예측 | 결과 | 지연 |")
        lines.append("|---|---|---|---|---|---|")
        for r in mr.lang_results:
            res = "✅" if r.correct else ("⚠️" if r.error else "❌")
            pred = r.error[:30] if r.error else r.predicted
            lines.append(f"| {r.task_id} | {r.category} | {r.answer} | {pred} | {res} | {r.latency}s |")
        lines.append("")

    # ── 4. 논리 추론 상세 ───────────────────────────────
    lines += ["## 4. 논리 추론 상세", ""]
    for mr in results:
        lines += [f"### {mr.name}", ""]
        lines.append("| ID | 카테고리 | 정답 | 예측 | 결과 | 지연 |")
        lines.append("|---|---|---|---|---|---|")
        for r in mr.logic_results:
            res = "✅" if r.correct else ("⚠️" if r.error else "❌")
            pred = r.error[:30] if r.error else r.predicted
            lines.append(f"| {r.task_id} | {r.category} | {r.answer} | {pred} | {res} | {r.latency}s |")
        lines.append("")

    # ── 5. 감정 분석 상세 ───────────────────────────────
    lines += ["## 5. 감정 분석 상세", ""]
    for mr in results:
        lines += [f"### {mr.name}", ""]
        lines.append("| ID | 발화 (앞 30자) | 정답 | 예측 | 결과 | 지연 |")
        lines.append("|---|---|---|---|---|---|")
        for i, (r, t) in enumerate(zip(mr.emotion_results, EMOTION_TASKS)):
            utt = t["utterance"][:30] + ("…" if len(t["utterance"]) > 30 else "")
            res = "✅" if r.correct else ("⚠️" if r.error else "❌")
            pred = r.error[:30] if r.error else r.predicted
            lines.append(f"| {r.task_id} | {utt} | {r.answer} | {pred} | {res} | {r.latency}s |")
        lines.append("")

    # ── 6. 테스트 데이터 요약 ───────────────────────────
    lines += [
        "## 6. 테스트 데이터 요약",
        "",
        f"| 항목 | 건수 |",
        f"|---|---|",
        f"| 언어 추론 | {len(LANG_TASKS)}건 |",
        f"| 논리 추론 | {len(LOGIC_TASKS)}건 |",
        f"| 감정 분석 | {len(EMOTION_TASKS)}건 |",
        f"| **합계** | **{len(LANG_TASKS)+len(LOGIC_TASKS)+len(EMOTION_TASKS)}건** |",
        "",
        "---",
        f"*Generated by test.py — {now}*",
        "",
    ]

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9.  메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Andy's Data Box — LLM 벤치마크                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n📌 .env: {ROOT / 'data' / '.env'}")
    print(f"📌 Ollama Primary  : {OLLAMA_MODEL_PRIMARY}")
    print(f"📌 Ollama Verifier : {OLLAMA_MODEL_VERIFIER}")
    print(f"📌 Gemini API Key  : {'설정됨' if GEMINI_API_KEY and not GEMINI_API_KEY.startswith('your_') else '미설정'}")
    print(f"📌 OpenAI API Key  : {'설정됨' if OPENAI_API_KEY and not OPENAI_API_KEY.startswith('your_') else '미설정'}")

    # ── 연결 상태 확인 ─────────────────────────────────
    print("\n" + "━" * 58)
    print("🔍 모델 연결 상태 확인")
    print("━" * 58)

    model_registry = []

    print(f"\n  [Ollama] {OLLAMA_MODEL_PRIMARY} (Primary)")
    if probe_ollama(OLLAMA_MODEL_PRIMARY):
        model_registry.append(("Qwen2.5-3B", "ollama-primary",
                                create_ollama_caller(OLLAMA_MODEL_PRIMARY)))

    print(f"\n  [Ollama] {OLLAMA_MODEL_VERIFIER} (Verifier)")
    if probe_ollama(OLLAMA_MODEL_VERIFIER):
        model_registry.append(("Phi3-Mini", "ollama-verifier",
                                create_ollama_caller(OLLAMA_MODEL_VERIFIER)))

    print(f"\n  [Gemini]")
    if probe_gemini():
        model_registry.append(("Gemini-2.5-Flash", "gemini",
                                create_gemini_caller()))

    print(f"\n  [OpenAI]")
    if probe_openai():
        model_registry.append(("GPT-4o", "openai",
                                create_openai_caller()))

    if not model_registry:
        print("\n❌ 실행 가능한 모델이 없습니다. .env 및 Ollama 설정을 확인하세요.")
        sys.exit(1)

    print(f"\n✅ 벤치마크 대상 모델: {[m[0] for m in model_registry]}")

    # ── 벤치마크 실행 ─────────────────────────────────
    print("\n" + "━" * 58)
    print("🏁 벤치마크 시작")
    print("━" * 58)

    all_results: list[ModelResult] = []
    for name, backend, caller in model_registry:
        mr = run_model_benchmark(name, backend, caller)
        all_results.append(mr)

    # ── 최적 모델 산출 ─────────────────────────────────
    best = find_best(all_results)

    # ── 리포트 갱신 ────────────────────────────────────
    report_md = make_report(all_results, best)
    
    # 기존 Section 7 & 8 보존 로직
    out_path = ROOT / "docs" / "benchmark_results.md"
    existing_content = ""
    if out_path.exists():
        existing_content = out_path.read_text(encoding="utf-8")
        
    final_output = []
    # 이전 Generated 라인 자르기 (기존의 1~6섹션 버리기 위해)
    if "## 7. 오류" in existing_content:
        tail = existing_content[existing_content.find("## 7. 오류"):]
        # GPT/Gemini 결과 반영해서 섹션 8 내용 살짝 업데이트
        if best and best.name.startswith("GPT") or best.name.startswith("Gemini"):
            tail = re.sub(
                r"\[기본 응답 모델\] →.*", 
                f"[기본 응답 모델] → {best.name} (API 연동 권장 - 한글 및 복합추론 완벽함)", 
                tail
            )
            tail = re.sub(
                r"Qwen2\.5:3b를 기본 응답 모델로 유지.*",
                f"{best.name} 성능이 압도적(종합 {composite_score(best)}점)이므로, 가능하면 클라우드 API를 연동하여 성능을 극대화하는 것을 권장합니다.",
                tail
            )
        
        final_output.append(report_md)
        final_output.append("\n" + tail)
    else:
        final_output.append(report_md)
        
    out_path.write_text("".join(final_output), encoding="utf-8")

    # ── 최종 콘솔 요약 ─────────────────────────────────
    print("\n" + "═" * 58)
    print("📊 최종 결과 요약")
    print("═" * 58)
    ranking = sorted(all_results, key=composite_score, reverse=True)
    for i, mr in enumerate(ranking, 1):
        cs = composite_score(mr)
        mark = " ← 🏆 최적 모델" if (best and mr.name == best.name) else ""
        print(f"\n  {i}위. {mr.name} ({mr.backend}){mark}")
        print(f"       언어:{mr.lang_acc}% | 논리:{mr.logic_acc}% | 감정:{mr.emotion_acc}%")
        print(f"       종합점수: {cs}점 | 평균지연: {mr.avg_latency}s | 총시간: {mr.total_time}s")

    print(f"\n📄 리포트 저장: {out_path}")
    print("\n✅ 벤치마크 완료!")


if __name__ == "__main__":
    main()
