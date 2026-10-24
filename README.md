# Jev as a Judge

Comparing Jev to other models for scoring RL rubrics

## Setup

The tasks are [WritingBench](https://github.com/X-PLUG/WritingBench), ported from Prime Intellect's
`prime-environments`: long-form writing queries (we use the ~470 English ones under 8000 chars),
each with its own 5-criterion, 1-10 rubric. Training is LoRA RL on [Tinker](https://thinkingmachines.ai/tinker/).

- `writing_bench.py`: the tasks (deterministic 80/20 train/test split) and the judges. A chat judge scores all 5 criteria in one JSON call; a decision judge (e.g. Jev) gets one `score` question per criterion through OpenRouter's Decisions API. Each judge's score is the mean criterion score in [0, 1].
- `judges.json`: the panel. `reward` is the judge whose score trains the policy; `judges` lists chat models, `decision_judges` lists decision models, and `reasoning_effort` applies to the OpenRouter chat judges. `anthropic/*` models go to the Anthropic API, everything else to OpenRouter.
- `train.py`: training rollouts are graded by the `reward` judge alone. At steps 0, 15 and 30, 20 test tasks are graded by the full panel. wandb gets reward, each judge's score, per-criterion scores, failure rate, latency and cost, plus truncation and reply length. `runs/<name>/` gets per-step HTML transcripts and rollout JSONL.

Keys in `.env`: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `TINKER_API_KEY`, `WANDB_API_KEY`.

```bash
uv sync
uv run --env-file .env train.py --help   # knobs: model, LoRA rank, LR, group sizes, steps, eval cadence
uv run --env-file .env train.py
```
