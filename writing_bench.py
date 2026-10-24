"""WritingBench tasks, plus a panel of LLM judges that grade replies against each task's rubric.

Each of WritingBench's 1000 queries (we use the English ones) ships five criteria with 1-10 level
descriptions. A judge scores a reply on all five in one call, and its score is the mean criterion
score in [0, 1]. Chat models get a JSON-scoring prompt; decision models (OpenRouter's Decisions
API, e.g. Jev) get one `score` question per criterion over its five level descriptions.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import httpx
from json_repair import repair_json
from openai import AsyncOpenAI

WRITING_BENCH_URL = (
    "https://raw.githubusercontent.com/X-PLUG/WritingBench/"
    "ae2d5176449b7b769815482641d35926f26793eb/benchmark_query/benchmark_all.jsonl"
)
CACHE_PATH = Path.home() / ".cache" / "writing_bench" / "benchmark_all.jsonl"
TEST_FRACTION = 0.2
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
LEVELS = ("1-2", "3-4", "5-6", "7-8", "9-10")

# Adapted from WritingBench's critic prompt (evaluate_benchmark.py) to grade all of a task's
# criteria in one call instead of one call per criterion.
JUDGE_PROMPT = """\
You are an expert evaluator with extensive experience in evaluating responses to writing queries.

Evaluate the Response to the Query on EACH of the Criteria below, following the Scoring Rules.

** Scoring Rules **
"1-2": Critical deficiencies and major issues that prevent adequate functionality.
"3-4": Lacking, with noticeable shortcomings that impact overall effectiveness and require improvement.
"5-6": Adequate but not exemplary; baseline performance that meets essential requirements. Most models may achieve this score.
"7-8": Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.
"9-10": Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws.

- Each criterion has its own level descriptions; use them to place the Response on that criterion's scale.
- Justify each score by citing specific strengths or deficiencies in the Response.
- Be very STRICT and do not be misled by format or length; evaluate beyond superficial appearances.
- Carefully discern whether the content of the Response is an illusion, appearing substantial but actually fabricated.
- A Response that only gives an introduction or overview without truly completing the Query is a failed response.
- Assign each criterion an integer score from 1 to 10.

** Query **
```
{query}
```

** Response **
```
{response}
```

** Criteria **
{criteria}

** Output format **
Return ONLY a JSON object, with one entry per criterion in the order listed above, and nothing else:
{"scores": [{"criterion": "<criterion name>", "reason": "<one or two sentences citing the Response>", "score": <integer 1-10>}, ...]}
"""

DECISION_INSTRUCTIONS = (
    "Score the Response to the Query on this criterion: {criterion}. Be very strict; do not be "
    "misled by format or length, watch for fabricated content, and treat a Response that only "
    "outlines or introduces without completing the Query as a failure."
)


@dataclass
class Task:
    id: str
    name: str
    prompt: str
    checklist: list[dict[str, str]]
    """Five criteria: `name`, `criteria_description`, and `1-2` ... `9-10` level texts."""


def load_tasks(split: str, max_query_chars: int | None = None) -> list[Task]:
    """English WritingBench tasks for `split` ("train", "test" or "all"), via a deterministic
    hash split on the task index so train and test stay disjoint."""
    if not CACHE_PATH.exists():
        response = httpx.get(WRITING_BENCH_URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(response.text, encoding="utf-8")
    tasks = []
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        digest = hashlib.sha256(f"writing-bench-{row['index']}".encode()).digest()
        is_test = int.from_bytes(digest[:8], "big") / 2**64 < TEST_FRACTION
        if (
            row["lang"] == "en"
            and (max_query_chars is None or len(row["query"]) <= max_query_chars)
            and (split == "all" or is_test == (split == "test"))
        ):
            tasks.append(Task(str(row["index"]), f"{row['domain1']} / {row['domain2']}", row["query"], row["checklist"]))
    return tasks


def render_criteria(checklist: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"{i}. {c['name']}: {c['criteria_description']}\n"
        + "\n".join(f"  {level}: {c[level]}" for level in LEVELS)
        for i, c in enumerate(checklist, 1)
    )


def parse_scores(text: str, num_criteria: int) -> list[int]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"judge returned no JSON object: {text[:200]!r}")
    entries = json.loads(repair_json(match.group(0))).get("scores", [])
    scores = [min(10, max(1, int(entry["score"]))) for entry in entries]
    if len(scores) != num_criteria:
        raise ValueError(f"judge scored {len(scores)} criteria, expected {num_criteria}")
    return scores


@dataclass
class Judge:
    name: str
    model: str
    weight: float = 0.0
    """1 for the judge whose score is the training reward, 0 for the comparison panel."""
    decision: bool = False
    base_url: str = OPENROUTER_URL
    api_key_var: str = "OPENROUTER_API_KEY"
    reasoning_effort: str | None = None

    @cached_property
    def client(self) -> AsyncOpenAI:
        # The SDK retries 429s, 5xxs and timeouts with backoff; the timeout keeps one stalled
        # read from hanging a whole training step.
        return AsyncOpenAI(
            base_url=self.base_url, api_key=os.environ[self.api_key_var], timeout=300, max_retries=4
        )

    async def grade(self, task: Task, reply: str) -> tuple[float, dict[str, float]]:
        """The reply's score in [0, 1] plus per-criterion, latency and cost metrics, keyed
        `<name>/...`. Raises once the client's retries are spent."""
        start = time.perf_counter()
        scores, metrics = await (self._decide if self.decision else self._chat)(task, reply)
        metrics |= {f"c{i}": s for i, s in enumerate(scores, 1)}
        # Wall-clock time of this judge's own calls, independent of how the trainer schedules
        # rollouts; concurrent calls can still queue at the provider.
        metrics["latency_s"] = time.perf_counter() - start
        return sum(scores) / len(scores), {f"{self.name}/{k}": v for k, v in metrics.items()}

    async def _chat(self, task: Task, reply: str) -> tuple[list[float], dict[str, float]]:
        fields = {"query": task.prompt, "response": reply, "criteria": render_criteria(task.checklist)}
        # One pass over the template, so a reply containing "{query}" isn't substituted again.
        prompt = re.sub(r"\{(query|response|criteria)\}", lambda m: fields[m.group(1)], JUDGE_PROMPT)
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        response = await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **extra
        )
        scores = parse_scores(response.choices[0].message.content or "", len(task.checklist))
        usage = response.usage
        metrics = {"output_tokens": float(usage.completion_tokens)} if usage else {}
        if usage and (cost := getattr(usage, "cost", None)) is not None:
            metrics["cost"] = float(cost)
        return [(s - 1) / 9 for s in scores], metrics

    async def _decide(self, task: Task, reply: str) -> tuple[list[float], dict[str, float]]:
        request = {
            "model": self.model,
            "state": {"query": task.prompt, "response": reply},
            "questions": {
                f"c{i}": {
                    "type": "score",
                    "instructions": DECISION_INSTRUCTIONS.format(criterion=f"{c['name']}: {c['criteria_description']}"),
                    "criteria": [f"{level}: {c[level]}" for level in LEVELS],
                }
                for i, c in enumerate(task.checklist, 1)
            },
        }
        raw = await self.client.post(DECISIONS_URL, cast_to=dict[str, Any], body=request)
        answers = [raw["answers"][f"c{i}"] for i in range(1, len(task.checklist) + 1)]
        metrics = {"confidence": sum(a["confidence"] for a in answers) / len(answers)}
        if (cost := (raw.get("usage") or {}).get("cost")) is not None:
            metrics["cost"] = float(cost)
        return [a["score"] / (len(LEVELS) - 1) for a in answers], metrics


def load_judges(path: str | Path, reward: str | None = None) -> list[Judge]:
    """judges.json lists OpenRouter model ids: chat models under `judges`, decision models under
    `decision_judges`, and a `reasoning_effort` for the OpenRouter chat judges (default "low";
    decision models don't reason, and Anthropic leaves extended thinking off unless asked).
    `anthropic/*` goes to Anthropic's own OpenAI-compatible API (`anthropic/claude-haiku-5.5` ->
    `claude-haiku-5-5`). Each judge is named by its id's last segment. `reward` (a judge name or
    model id) overrides the file's `reward`."""
    spec = json.loads(Path(path).read_text())
    chat, decision = spec.get("judges", []), spec.get("decision_judges", [])
    reward = reward or spec["reward"]
    judges = []
    for slug in chat + decision:
        name = re.sub(r"\W+", "_", slug.split("/")[-1])
        judge = Judge(name=name, model=slug, weight=float(reward in (slug, name)))
        if slug in decision:
            judge.decision = True
        elif slug.startswith("anthropic/"):
            judge.model = slug.removeprefix("anthropic/").replace(".", "-")
            judge.base_url, judge.api_key_var = "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"
        else:
            judge.reasoning_effort = spec.get("reasoning_effort", "low")
        judges.append(judge)
    if not any(j.weight for j in judges):
        raise ValueError(f"reward judge {reward!r} is not in `judges` or `decision_judges`")
    return judges
