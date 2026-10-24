"""LoRA RL on WritingBench with Tinker, using judges.json's `reward` judge (Jev) as the reward.

Training rollouts are graded by the reward judge alone. Every `eval_every` steps (including step
0, the base-model baseline), `n_eval` held-out test tasks are graded by the full judges.json panel,
so per-judge scores, failure rates, latency and spend land in wandb beside the training curves.
Tinker also writes per-step HTML transcripts and rollout JSONL under `log_path`.

    uv run --env-file .env train.py                                  # judges.json's reward
    uv run --env-file .env train.py rewards=jev_latest,mercury_3     # one run per judge, in parallel
"""

import asyncio
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import chz
import tinker
from tinker_cookbook import cli_utils, renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from writing_bench import Judge, Task, load_judges, load_tasks


class WritingEnv(Env):
    def __init__(self, renderer: renderers.Renderer, task: Task, judges: list[Judge], is_train: bool):
        self.renderer, self.task, self.judges, self.is_train = renderer, task, judges, is_train
        self.convo = [{"role": "user", "content": task.prompt}]

    async def initial_observation(self):
        return self.renderer.build_generation_prompt(self.convo), self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        # GPT-OSS sometimes ends an answer with <|end|> instead of <|return|> and carries on
        # (more analysis, another final answer); judge only its last answer, not the drafts
        # glued together.
        answers = [p["text"] for p in message["content"] if p["type"] == "text"] \
            if isinstance(message["content"], list) else [message["content"]]
        # A reply cut off by max_tokens (often mid-reasoning) would hand the judges raw analysis
        # text; it scores 0 without any judge call, and stays out of the judges' means.
        reply = answers[-1] if termination.is_clean and answers else ""
        metrics = {
            "truncated": float(not reply),
            "reply_chars": float(len(reply)),
            "multi_answer": float(len(answers) > 1),
        }
        reward, scores = 0.0, {}
        if reply:
            results = await asyncio.gather(
                *(judge.grade(self.task, reply) for judge in self.judges), return_exceptions=True
            )
            for judge, result in zip(self.judges, results):
                if isinstance(result, BaseException):
                    # Training on a failed reward call as 0 would push the policy away from a
                    # possibly good reply; raising lets RetryOnFailure redo or drop the rollout.
                    if self.is_train and judge.weight:
                        raise result
                    metrics[f"judge/{judge.name}/failed"] = 1.0
                    continue
                score, judge_metrics = result
                scores[judge.name] = score
                reward += judge.weight * score
                metrics |= {f"judge/{judge.name}": score, f"judge/{judge.name}/failed": 0.0}
                metrics |= {f"judge/{k}": v for k, v in judge_metrics.items()}
            # Every policy looks good to the judge it trained on; compare runs on the rest.
            heldout = [scores[j.name] for j in self.judges if not j.weight and j.name in scores]
            if heldout:
                metrics["judge/heldout_mean"] = sum(heldout) / len(heldout)

        with logtree.scope_header(f"Task {self.task.id}: {self.task.name}"):
            logtree.log_text(self.task.prompt)
        with logtree.scope_header("Policy reply"):
            logtree.log_text(reply or "(truncated)")
        with logtree.scope_header("Judges"):
            logtree.table_from_dict(
                {name: f"{s:.3f}" for name, s in scores.items()} | {"reward": f"{reward:.3f}"},
                caption="Mean criterion score per judge, in [0, 1]",
            )
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )


class WritingDataset(RLDataset):
    def __init__(self, envs: list[partial], group_size: int, groups_per_batch: int):
        self.envs, self.group_size, self.groups_per_batch = envs, group_size, groups_per_batch

    def __len__(self) -> int:
        return len(self.envs) // self.groups_per_batch

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch = self.envs[index * self.groups_per_batch : (index + 1) * self.groups_per_batch]
        return [ProblemGroupBuilder(env, self.group_size, "writing_bench") for env in batch]


@chz.chz
class WritingDatasetBuilder(RLDatasetBuilder):
    model_name: str
    renderer_name: str
    group_size: int
    groups_per_batch: int
    n_eval: int
    max_query_chars: int
    judges_file: str
    reward: str
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        renderer = renderers.get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        panel = load_judges(self.judges_file, self.reward)
        train_tasks, test_tasks = load_tasks("train", self.max_query_chars), load_tasks("test", self.max_query_chars)
        random.Random(self.seed).shuffle(train_tasks)
        random.Random(self.seed).shuffle(test_tasks)
        reward_judges = [j for j in panel if j.weight]
        train_set = WritingDataset(
            [partial(WritingEnv, renderer, t, reward_judges, True) for t in train_tasks],
            self.group_size,
            self.groups_per_batch,
        )
        test_envs = [partial(WritingEnv, renderer, t, panel, False) for t in test_tasks[: self.n_eval]]
        return train_set, WritingDataset(test_envs, group_size=1, groups_per_batch=len(test_envs))


@chz.chz
class CLIConfig:
    model_name: str = "openai/gpt-oss-20b"
    renderer_name: str = "gpt_oss_low_reasoning"
    lora_rank: int = 32
    learning_rate: float = 4e-5
    """Not calibrated for GPT-OSS in tinker_cookbook.hyperparam_utils; a starting guess."""

    group_size: int = 8
    groups_per_batch: int = 4
    max_tokens: int = 8192
    """WritingBench asks for ~2000 words at the median (~2.7k tokens), plus reasoning."""
    temperature: float = 1.0
    max_steps: int = 31
    """Evals run at the start of a step, so 31 steps puts the last eval (step 30) on 30 updates."""
    max_query_chars: int = 8000

    judges_file: str = "judges.json"
    rewards: str = ""
    """Comma-separated judge names (or "all") to each train a policy on, in parallel, one run
    and wandb run per judge; empty uses judges.json's `reward`."""
    n_eval: int = 20
    eval_every: int = 15
    """Each eval runs the full panel on `n_eval` tasks, so this is the main judge-spend knob."""
    save_every: int = 15

    log_path: str | None = None
    wandb_project: str | None = "jev-as-a-judge"
    wandb_name: str | None = None


async def main(cli: CLIConfig) -> None:
    reward = next(j.name for j in load_judges(cli.judges_file, cli.rewards or None) if j.weight)
    run_name = cli.wandb_name or (
        f"{reward}-{cli.model_name.split('/')[-1]}-r{cli.lora_rank}-lr{cli.learning_rate}"
        f"-g{cli.groups_per_batch}x{cli.group_size}-{time.strftime('%m%d-%H%M')}"
    )
    log_path = cli.log_path or str(Path("runs") / run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
    config = train.Config(
        model_name=cli.model_name,
        renderer_name=cli.renderer_name,
        recipe_name="writing_bench",
        lora_rank=cli.lora_rank,
        learning_rate=cli.learning_rate,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        max_steps=cli.max_steps,
        dataset_builder=WritingDatasetBuilder(
            model_name=cli.model_name,
            renderer_name=cli.renderer_name,
            group_size=cli.group_size,
            groups_per_batch=cli.groups_per_batch,
            n_eval=cli.n_eval,
            max_query_chars=cli.max_query_chars,
            judges_file=cli.judges_file,
            reward=reward,
        ),
        # A reward-judge failure raises; retry the rollout, or skip the group once the budget is spent.
        rollout_error_tolerance=RetryOnFailure(max_retries=4, per_rollout_timeout=1800),
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        log_path=log_path,
        wandb_project=cli.wandb_project,
        wandb_name=run_name,
    )
    await train.main(config)


def launch_parallel(rewards: list[str]) -> None:
    """One `train.py rewards=<judge>` process per judge (Tinker trains remotely, so these only
    wait on the network), each logging to runs/parallel-<time>/<judge>.log and its own wandb run."""
    log_dir = Path("runs") / f"parallel-{time.strftime('%m%d-%H%M')}"
    log_dir.mkdir(parents=True)
    args = [a for a in sys.argv[1:] if not a.startswith("rewards=")]
    procs = {}
    for reward in rewards:
        with open(log_dir / f"{reward}.log", "w") as log:
            procs[reward] = subprocess.Popen(
                [sys.executable, __file__, *args, f"rewards={reward}"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
    print(f"training {len(procs)} policies in parallel; logs in {log_dir}/")
    for reward, proc in procs.items():
        print(f"{reward}: exit {proc.wait()}")


if __name__ == "__main__":
    cli = chz.entrypoint(CLIConfig)
    rewards = [r for r in cli.rewards.split(",") if r]
    if rewards == ["all"]:
        rewards = [j.name for j in load_judges(cli.judges_file)]
    if len(rewards) > 1:
        launch_parallel(rewards)
    else:
        asyncio.run(main(cli))
