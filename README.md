# Jev as a Judge

Comparing Jev to other models for scoring RL rubrics

## Setup

The task is [HumanAgencyBench](https://arxiv.org/abs/2509.08494) (via Prime Intellect's
`agency_bench` env): 3,000 short-answer prompts across six dimensions of supporting a user's agency,
each graded by a rubric of issues with point deductions. Its dataset also ships 75,000 replies from
25 models graded by o3, which we use to check judges against o3 before training.

- `agency_bench.py`: tasks (deterministic 80/20 train/test split), judges, and `check`. Chat judges get the benchmark's own judge prompt; decision models such as Jev get one yes/no question per issue through OpenRouter's Decisions API and score the expected deduction.
- `rubrics.json`: the six rubrics (issues, deductions, judge prompts), parsed from Prime's port.
- `judges.json`: the judge panel. `reward` is the default reward judge; `judges` lists chat models, `decision_judges` decision models; `anthropic/*` goes to the Anthropic API, the rest to OpenRouter.
- `train.py`: GRPO with LoRA on vLLM rollouts (ported from rl-experiments), one reward judge per run, each run on its own GPU (vLLM sleeps while the trainer runs). Held-out evals graded by the full panel at step 0 and every 25 steps; rollouts, eval replies with every judge's scores, metrics and checkpoints are saved under `runs/` and `checkpoints/`.

Keys in `.env`: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `WANDB_API_KEY`.

```bash
uv run --env-file .env agency_bench.py check --prompts 60 --replies 8   # judges vs o3 (laptop is fine)
uv sync --extra train                                                     # GPU box only
uv run --env-file .env train.py --rewards jev_latest,mercury_3            # one run per GPU
```
