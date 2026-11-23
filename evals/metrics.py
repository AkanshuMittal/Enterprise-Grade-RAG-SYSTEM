import os
import asyncio
import logfire
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)
import time
import json
import hashlib
from types import SimpleNamespace
from typing import List, Literal

import openai
from pydantic import BaseModel

load_dotenv()  # load JUDGE_GROQ_API_KEY from .env

JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "qwen/qwen3.8-27b"
JUDGE_MAX_TOKENS = 2048      # Groq counts the requested max_tokens toward the TPM limit, so keep it modest
COOLDOWN_STANDARD = 20      # between metrics, lets the per-minute token window recover
COOLDOWN_MINI = 5       # between individual samples
GENERAL_BATCH_SIZE = 1   # one sample at a time
CONTEXT_TRUNCATE = 2000  # safety cap per chunk, above the longest measured chunk
CONTEXT_LIMIT = 5        # same number of chunks the generator sees (rerank 15 -> 5)

JUDGE_MIN_INTERVAL = 6.0    # seconds between judge calls (Groq limit is tokens/minute, not requests)


def _throttle_client(client, min_interval: float = JUDGE_MIN_INTERVAL):
    """Space judge calls apart and retry on 429 instead of crashing the eval.
    Must be applied to the raw client BEFORE it is passed to llm_factory()."""
    original_create = client.chat.completions.create
    lock = asyncio.Lock()
    state = {"last": 0.0}

    async def throttled_create(*args, **kwargs):
        # Qwen3 is a reasoning model. Turn its thinking text off so RAGAS gets clean JSON.
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("reasoning_effort", "none")
        kwargs["extra_body"] = extra
        for attempt in range(5):
            async with lock:
                wait = min_interval - (time.monotonic() - state["last"])
                if wait > 0:
                    await asyncio.sleep(wait)
                state["last"] = time.monotonic()
            try:
                return await original_create(*args, **kwargs)
            except openai.RateLimitError as e:
                if "tokens per day" in str(e).lower() or "TPD" in str(e):
                    print("\n*** Judge DAILY token quota is used up. Waiting will not help for hours. ***")
                    print("*** Finished metrics are already saved as evals/results_<metric>.csv.")
                    print("*** Re-run later with:  python verification/run_phase2.py --resume\n")
                    raise
                if attempt == 4:
                    raise
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get("retry-after"))
                except Exception:
                    pass
                pause = min(max(retry_after or 30, 5), 90) + 2
                print(f"   (judge 429, pausing {pause:.0f}s, retry {attempt + 1}/5: {str(e)[:80]}...)")
                await asyncio.sleep(pause)

    client.chat.completions.create = throttled_create
    return client


def _build_judge():
    api_key = os.getenv("JUDGE_GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("JUDGE_GROQ_API_KEY is missing. Add it to .env (no quotes, no spaces around '=').")
    client = AsyncOpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    _throttle_client(client)
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client, max_tokens=JUDGE_MAX_TOKENS)
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings


async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (rate-limit buffer)..."
    if status_cb:
        status_cb(msg)
    await asyncio.sleep(seconds)
    if status_cb:
        status_cb("✅ Ready — starting next experiment.")


ANSWER_CORRECTNESS_MODE = "scoped"
SCOPED_PROMPT = """You are grading an answer to a user's question against a reference answer.

QUESTION:
{question}

REFERENCE ANSWER (trusted):
{reference}

ANSWER TO GRADE:
{response}

Step 1. From the REFERENCE ANSWER only, list the facts that are REQUIRED to answer the QUESTION
exactly as it was asked. Keep each fact short and atomic. Do NOT list reference facts that go
beyond what the question asks. Do NOT invent facts that are not in the reference.

Step 2. For each required fact, set status to one of:
- "covered": the ANSWER states this fact correctly (different wording is fine).
- "missing": the ANSWER does not state it.
- "contradicted": the ANSWER states something that conflicts with it.

Rules:
- Extra information in the ANSWER that the question did not ask for must NOT lower the result,
  unless it contradicts a required fact.
- Judge meaning, not exact wording. Commands, flags, names and values must match exactly.
- Return only the structured result."""


class _FactCheck(BaseModel):
    fact: str
    status: Literal["covered", "missing", "contradicted"]


class _ScopedVerdict(BaseModel):
    required_facts: List[_FactCheck]


class ScopedCorrectness:
    """score = covered required facts / all required facts (computed here, not by the LLM)."""

    def __init__(self, llm):
        self.llm = llm
        self.details = {}          # question -> list of {fact, status}, saved for hand inspection

    async def ascore(self, user_input: str, response: str, reference: str):
        prompt = SCOPED_PROMPT.format(question=user_input, reference=reference, response=response)
        verdict = await self.llm.agenerate(prompt, _ScopedVerdict)
        facts = verdict.required_facts
        self.details[user_input] = [f.model_dump() for f in facts]
        if not facts:
            return SimpleNamespace(value=0.0)          # judge returned nothing usable: do not give free credit
        covered = sum(1 for f in facts if f.status == "covered")
        return SimpleNamespace(value=covered / len(facts))

    async def abatch_score(self, inputs: list):
        return [await self.ascore(**inp) for inp in inputs]


def _save_scoped_details(metric: "ScopedCorrectness") -> None:
    path = os.path.join(os.path.dirname(__file__), "results_answer_correctness_details.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metric.details, f, indent=2, ensure_ascii=False)


RESUME = False   # set by run_phase2.py --resume: reuse metric CSVs that already exist
SKIPPED = []   # samples we could not score, kept so we can report them honestly

def _prep_samples(golden_dataset: dict) -> list:
    valid = []
    SKIPPED.clear()
    for s in golden_dataset["rag_samples"]:
        response = (s.get("actual_response") or "").strip()
        raw_contexts = s.get("actual_contexts") or []
        if not response:
            SKIPPED.append((s["question"][:60], "no response"))
            continue
        if not raw_contexts:
            SKIPPED.append((s["question"][:60], "no contexts (blocked or retrieval failed)"))
            continue
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    df = pd.DataFrame([
        {"question": s["question"][:65], metric_key: round(float(r.value), 3)}
        for s, r in zip(samples, scores)
    ])
    # Save each metric as soon as it finishes, so a later crash loses nothing.
    out_path = os.path.join(os.path.dirname(__file__), f"results_{metric_key}.csv")
    df.to_csv(out_path, index=False)
    with open(_sig_path(metric_key), "w", encoding="utf-8") as f:
        f.write(_signature(samples, metric_key))
    return df


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Keeps bursts inside the judge's tokens-per-minute limit.
    """
    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        scores = await metric.abatch_score(batch)
        all_scores.extend(scores)
    return all_scores

def _signature(samples: list, metric_key: str = "") -> str:
    """Fingerprint of everything a score depends on: the answers, contexts, references, and (for
    answer_correctness) which definition is used. If any of these change, the saved score is stale."""
    extra = ANSWER_CORRECTNESS_MODE if metric_key == "answer_correctness" else ""
    payload = [(s["question"], s["actual_response"], s["actual_contexts"], s.get("reference", "")) for s in samples]
    payload.append(extra)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


# Metrics whose saved CSV is only reused if it carries a matching signature file. An old
# answer_correctness CSV (standard RAGAS, no signature) must never be reused for the custom version.
STRICT_SIG_KEYS = {"answer_correctness"}


def _sig_path(metric_key: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"results_{metric_key}.sig")


def _try_resume(metric_key: str, samples: list):
    """Return a saved DataFrame if RESUME is on and the CSV still matches these samples, else None."""
    if not RESUME:
        return None
    path = os.path.join(os.path.dirname(__file__), f"results_{metric_key}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) != len(samples) or list(df["question"]) != [s["question"][:65] for s in samples]:
        return None
    sig_file = _sig_path(metric_key)
    if os.path.exists(sig_file):
        with open(sig_file, encoding="utf-8") as f:
            if f.read().strip() != _signature(samples, metric_key):
                print(f"   (saved {metric_key} is from DIFFERENT answers/contexts/definition, re-running it)")
                return None
    elif metric_key in STRICT_SIG_KEYS:
        print(f"   (saved {metric_key} has no signature: it may be an older definition, re-running it)")
        return None
    else:
        print(f"   (saved {metric_key} has no signature file: assuming it matches the current Phase 1 data)")
    return df


async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all experiments (5 LLM-judged + Tool Correctness). Returns dict keyed by metric name -> DataFrame.
    With RESUME on, metrics whose CSV already exists are loaded instead of re-run.
    status_cb(message: str) is called for live UI updates.
    """
    samples = _prep_samples(golden_dataset)

    if SKIPPED:
        print(f"WARNING: skipped {len(SKIPPED)} of {len(golden_dataset['rag_samples'])} samples:")
        for q, why in SKIPPED:
            print("   -", q, "|", why)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    judge_llm, ragas_embeddings = _build_judge()
    say = status_cb or (lambda m: None)
    scoped_metric = ScopedCorrectness(judge_llm)

    def q_resp(s):  return {"user_input": s["question"], "response": s["actual_response"]}
    def q_ctx_ref(s): return {"user_input": s["question"], "reference": s["reference"],
                              "retrieved_contexts": s["actual_contexts"]}

    specs = [
        ("faithfulness", "Faithfulness", lambda: Faithfulness(llm=judge_llm),
         lambda s: {"user_input": s["question"], "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"]}),
        ("answer_relevancy", "Answer Relevancy",
         lambda: AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings), q_resp),
        ("context_precision", "Context Precision", lambda: ContextPrecision(llm=judge_llm), q_ctx_ref),
        ("context_recall", "Context Recall", lambda: ContextRecall(llm=judge_llm), q_ctx_ref),
        ("answer_correctness",
         "Answer Correctness (custom, question-scoped)" if ANSWER_CORRECTNESS_MODE == "scoped"
         else "Answer Correctness (standard RAGAS)",
         (lambda: scoped_metric) if ANSWER_CORRECTNESS_MODE == "scoped"
         else (lambda: AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings)),
         lambda s: {"user_input": s["question"], "response": s["actual_response"], "reference": s["reference"]}),
    ]

    results = {}
    with logfire.span("Eval Phase 2 - All Metrics", total_samples=len(samples)):
        for n, (key, label, make_metric, make_input) in enumerate(specs, start=1):
            saved = _try_resume(key, samples)
            if saved is not None:
                say(f"Exp {n}/{len(specs) + 1} - {label}: loaded saved results (resume), skipping.")
                results[key] = saved
                continue
            say(f"Exp {n}/{len(specs) + 1} - {label} ({len(samples)} samples)...")
            with logfire.span(f"Exp {n} - {label}"):
                inputs = [make_input(s) for s in samples]
                scores = await _batched_score(make_metric(), inputs, samples, status_cb, label)
                df = _score_df(key, samples, scores)
                results[key] = df
                logfire.info(f"{label} done", avg=round(float(df[key].mean()), 3))
                if key == "answer_correctness" and ANSWER_CORRECTNESS_MODE == "scoped":
                    _save_scoped_details(scoped_metric)
            await _cooldown(COOLDOWN_STANDARD, label, status_cb)

        # Exp 6: Tool Correctness (no LLM, Jaccard)
        say(f"Exp {len(specs) + 1}/{len(specs) + 1} - Tool Correctness (zero LLM calls)...")
        rows = []
        for s in samples:
            called = set(s.get("actual_tools_called") or [])
            expected = set(s.get("expected_tools") or [])
            union = len(called | expected)
            rows.append({"question": s["question"][:65],
                         "tool_correctness": round(len(called & expected) / union if union else 0.0, 3)})
        results["tool_correctness"] = pd.DataFrame(rows)
        say("All experiments complete!")

    return results