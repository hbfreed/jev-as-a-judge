"""GRPO with LoRA on HumanAgencyBench, with one reward judge per run and rollouts on vLLM.

Ported from rl-experiments/train.py: vLLM samples each GRPO group, a PEFT LoRA trainer updates the
policy, and each step's adapter reaches vLLM through tmpfs and an in-place LoRARequest reload
(~0.5 s, no base-weight copy). vLLM and the trainer share one GPU: the engine sleeps (weights
offloaded to CPU, KV cache freed) while the trainer runs, so each GPU hosts one whole run.

Defaults follow Prime Intellect's prime-rl examples for Qwen3-4B-Instruct-2507 (LoRA, lr 1e-5,
16 prompts x 16 rollouts, 200 steps). Every `eval_every` steps, starting at step 0, held-out
prompts are answered greedily and graded by the whole judges.json panel.

    uv run --env-file .env train.py --reward jev_latest                 # one run
    uv run --env-file .env train.py --rewards jev_latest,mercury_3      # one run per GPU
"""

import argparse
import asyncio
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from agency_bench import RUBRICS, Judge, Task, load_judges, load_tasks


@dataclass
class Config:
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    reward: str = "jev_latest"
    """Judge name (or model id) from judges.json whose score is the reward."""
    judges_file: str = "judges.json"

    # LoRA on every linear layer ("LoRA Without Regret"). Rank barely matters for RL; 8 matches
    # Prime's wiki-search example for this model. alpha = 2 * rank keeps alpha/rank fixed.
    lora_rank: int = 8
    lora_alpha: int = 16
    lr: float = 1e-5
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    kl_coef: float = 0.05
    """Per-token k3 KL(policy || base), the base being this model with the adapter off; 0 disables."""

    num_steps: int = 200
    prompts_per_step: int = 16
    group_size: int = 16
    max_new_tokens: int = 512
    temperature: float = 1.0
    mask_truncated: bool = True
    """Zero the advantage of replies cut off by max_new_tokens (they stay in the group baseline)."""
    micro_batch_tokens: int = 8192
    judge_concurrency: int = 32

    eval_every: int = 25
    eval_n: int = 120
    """Held-out prompts per eval, balanced across the six dimensions."""

    vllm_gpu_memory_utilization: float = 0.5
    """vLLM's share of the GPU while awake; the trainer's weights stay resident beside it."""
    ckpt_every: int = 25
    seed: int = 1223
    use_wandb: bool = True
    wandb_project: str = "jev-as-a-judge"


# --- policy and engine -------------------------------------------------------------------------


def load_policy(cfg: Config):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    ))
    # With LoRA the embeddings are frozen, so inputs must require grad for checkpointed blocks to
    # have a grad path; and HF only checkpoints in train mode.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    return model.to("cuda"), tokenizer


def build_engine(cfg: Config):
    import vllm

    return vllm.LLM(
        model=cfg.model,
        dtype="bfloat16",
        max_model_len=2048,  # prompts ~250 tokens + max_new_tokens
        max_num_seqs=256,  # one decode wave for a 16 x 16 step
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=1 << max(3, (cfg.lora_rank - 1).bit_length()),  # vLLM wants a power of 2 >= 8
        max_loras=1,
        enable_sleep_mode=True,
    )


def sync_adapter(model, adapter_dir: str):
    """Publish the policy by writing only the LoRA adapter to tmpfs; load_inplace makes vLLM reload
    it under the same id on the next generate (a same-id request without it keeps the stale one)."""
    from vllm.lora.request import LoRARequest

    model.save_pretrained(adapter_dir)
    return LoRARequest(lora_name="policy", lora_int_id=1, lora_path=adapter_dir, load_inplace=True)


def generate(llm, tasks: list[Task], n: int, temperature: float, cfg: Config, lora_request):
    """Wake the engine, sample `n` replies per task with the current adapter, and put it back to
    sleep so the trainer has the GPU."""
    import vllm

    llm.wake_up()
    try:
        return llm.chat(
            [t.messages for t in tasks],
            sampling_params=vllm.SamplingParams(n=n, temperature=temperature, max_tokens=cfg.max_new_tokens),
            lora_request=lora_request,
            use_tqdm=False,
        )
    finally:
        llm.sleep(level=1)


async def grade_all(judge: Judge, pairs: list[tuple[Task, str]], concurrency: int):
    """(score, metrics) per (task, reply), or None where the judge failed after its retries."""
    sem = asyncio.Semaphore(concurrency)

    async def one(task, reply):
        async with sem:
            try:
                return await judge.grade(task, reply)
            except Exception as e:
                print(f"  [judge] {judge.name} failed: {e!r}"[:300], flush=True)
                return None

    return await asyncio.gather(*(one(t, r) for t, r in pairs))


# --- GRPO (from rl-experiments/train.py) -------------------------------------------------------


def group_relative_advantages(rewards, num_prompts, group_size, eps=1e-6):
    """(r - mean) / (std + eps) within each prompt's group; tied groups get exactly 0."""
    grouped = rewards.view(num_prompts, group_size)
    advantages = (grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + eps)
    tied = grouped.max(1, keepdim=True).values == grouped.min(1, keepdim=True).values
    return torch.where(tied, 0.0, advantages).reshape(-1)


def iter_micro_batches(rows, lengths, budget_tokens):
    """Pack rows, longest first, into micro-batches whose padded size stays under budget_tokens."""
    batch: list[int] = []
    for k in sorted(rows, key=lengths.__getitem__, reverse=True):
        if batch and (len(batch) + 1) * lengths[batch[0]] > budget_tokens:
            yield batch
            batch = []
        batch.append(k)
    if batch:
        yield batch


def build_batch(prompt_ids, completion_ids, pad_id):
    seqs = [p + c for p, c in zip(prompt_ids, completion_ids)]
    n, T = len(seqs), max(len(s) for s in seqs)
    input_ids = torch.full((n, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, T), dtype=torch.long)
    completion_mask = torch.zeros((n, T), dtype=torch.long)
    for k, (p, seq) in enumerate(zip(prompt_ids, seqs)):
        input_ids[k, : len(seq)] = torch.tensor(seq)
        attention_mask[k, : len(seq)] = 1
        completion_mask[k, len(p) : len(seq)] = 1
    return input_ids.cuda(), attention_mask.cuda(), completion_mask.cuda()


def grpo_loss(model, input_ids, attention_mask, completion_mask, advantages, normalizer, kl_coef, kl_normalizer):
    """On-policy policy-gradient loss for one micro-batch, normalized by the step's trained-token
    count, plus the k3 KL anchor. Also returns detached sums of sampled-token logprobs (for an
    entropy estimate) and token KL."""
    targets = input_ids[:, 1:]
    ref_logprobs = None
    if kl_coef > 0:
        with torch.no_grad(), model.disable_adapter():
            ref_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
            ref_logprobs = -F.cross_entropy(ref_logits.transpose(1, 2), targets, reduction="none")
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    token_logprobs = -F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    mask = completion_mask[:, 1:].to(token_logprobs.dtype)
    loss = -(advantages[:, None] * token_logprobs * mask).sum() / normalizer
    kl_sum = torch.zeros((), device=input_ids.device)
    if kl_coef > 0:
        # Mask before exp, so junk logits at prompt/pad positions can't overflow into inf * 0.
        delta = (ref_logprobs - token_logprobs).float() * completion_mask[:, 1:]
        token_kl = delta.exp() - delta - 1.0
        loss = loss + kl_coef * token_kl.sum() / kl_normalizer
        kl_sum = token_kl.sum().detach()
    return loss, (token_logprobs.detach() * mask).sum(), kl_sum


def iter_prompt_batches(tasks: list[Task], batch_size: int, seed: int):
    """Endless batches without replacement within an epoch, reshuffled each epoch."""
    rng = random.Random(seed)
    order = list(range(len(tasks)))
    rng.shuffle(order)
    pos = 0
    while True:
        if pos + batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        yield [tasks[i] for i in order[pos : pos + batch_size]]
        pos += batch_size


# --- eval ----------------------------------------------------------------------------------------


def eval_tasks(cfg: Config) -> list[Task]:
    test = load_tasks("test")
    random.Random(cfg.seed).shuffle(test)
    per_dim = cfg.eval_n // len(RUBRICS)
    return [t for dim in sorted(RUBRICS) for t in [t for t in test if t.dimension == dim][:per_dim]]


def evaluate(step, llm, panel, tasks, cfg, lora_request, loop, out_dir: Path) -> dict:
    """Greedy replies to held-out prompts, graded by every judge. Saves each reply with all judge
    scores to eval_step_<step>.jsonl and returns eval/* metrics."""
    t = time.perf_counter()
    outs = generate(llm, tasks, 1, 0.0, cfg, lora_request)
    replies = [o.outputs[0].text for o in outs]
    async def grade_panel():  # gather inside a coroutine, so its future belongs to `loop`
        return await asyncio.gather(*(
            grade_all(judge, list(zip(tasks, replies)), cfg.judge_concurrency) for judge in panel
        ))

    results = loop.run_until_complete(grade_panel())
    metrics = {
        "eval/trunc_rate": statistics.mean(o.outputs[0].finish_reason == "length" for o in outs),
        "eval/mean_len": statistics.mean(len(o.outputs[0].token_ids) for o in outs),
    }
    for judge, res in zip(panel, results):
        ok = [r for r in res if r is not None]
        metrics[f"eval/{judge.name}/failed"] = 1 - len(ok) / len(res)
        if ok:
            metrics[f"eval/{judge.name}"] = statistics.mean(r[0] for r in ok)
            metrics[f"eval/{judge.name}/latency_s"] = statistics.median(r[1][f"{judge.name}/latency_s"] for r in ok)
    # Every policy looks good to the judge it trained on; compare runs on the rest of the panel.
    heldout = [metrics[f"eval/{j.name}"] for j in panel if not j.weight and f"eval/{j.name}" in metrics]
    if heldout:
        metrics["eval/heldout_mean"] = statistics.mean(heldout)
    with (out_dir / f"eval_step_{step}.jsonl").open("w") as f:
        for i, (task, reply) in enumerate(zip(tasks, replies)):
            f.write(json.dumps({
                "task_id": task.id, "dimension": task.dimension, "reply": reply,
                "judges": {j.name: (None if res[i] is None else {"score": res[i][0], **res[i][1]})
                           for j, res in zip(panel, results)},
            }) + "\n")
    metrics["eval/eval_s"] = time.perf_counter() - t
    print(f"  [eval] step {step}: " + " | ".join(
        f"{j.name} {metrics.get(f'eval/{j.name}', float('nan')):.3f}" for j in panel
    ) + f" | heldout {metrics.get('eval/heldout_mean', float('nan')):.3f} | len {metrics['eval/mean_len']:.0f}", flush=True)
    return metrics


# --- training loop -----------------------------------------------------------------------------


def train(cfg: Config) -> None:
    import bitsandbytes as bnb
    from transformers import get_constant_schedule_with_warmup

    panel = load_judges(cfg.judges_file, cfg.reward)
    reward_judge = next(j for j in panel if j.weight)
    run_name = (
        f"{reward_judge.name}-{cfg.model.split('/')[-1].lower()}-r{cfg.lora_rank}-lr{cfg.lr:g}"
        f"-{time.strftime('%m%d-%H%M')}"
    )
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg) | {"run_name": run_name}, indent=2))

    train_tasks = load_tasks("train")
    held_out = eval_tasks(cfg)
    print(f"[data] {len(train_tasks)} train prompts, {len(held_out)} eval prompts; reward = {reward_judge.name}", flush=True)

    # Build the engine first, while the GPU is empty, then sleep it so the trainer can load.
    llm = build_engine(cfg)
    llm.sleep(level=1)
    print("[ready] vLLM engine up", flush=True)
    model, tokenizer = load_policy(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps)
    adapter_dir = f"/dev/shm/jev-as-a-judge/{run_name}"
    lora_request = sync_adapter(model, adapter_dir)
    # One loop for the whole run: the judges' HTTP clients are bound to the loop they first ran on.
    loop = asyncio.new_event_loop()

    if cfg.use_wandb:
        import wandb

        wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg))

    def log(metrics: dict, step: int) -> None:
        if cfg.use_wandb:
            wandb.log(metrics, step=step)
        with (out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    prompt_batches = iter_prompt_batches(train_tasks, cfg.prompts_per_step, cfg.seed)
    for step in range(cfg.num_steps + 1):
        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.num_steps):
            log(evaluate(step, llm, panel, held_out, cfg, lora_request, loop, out_dir), step)
        if step == cfg.num_steps:
            break

        t_step = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        batch = next(prompt_batches)
        t = time.perf_counter()
        outs = generate(llm, batch, cfg.group_size, cfg.temperature, cfg, lora_request)
        gen_s = time.perf_counter() - t

        flat = [(task, out, c) for task, out in zip(batch, outs) for c in out.outputs]
        t = time.perf_counter()
        graded = loop.run_until_complete(grade_all(reward_judge, [(task, c.text) for task, _, c in flat], cfg.judge_concurrency))
        score_s = time.perf_counter() - t

        prompt_ids = [list(out.prompt_token_ids) for _, out, _ in flat]
        completion_ids = [list(c.token_ids) for _, _, c in flat]
        truncated = torch.tensor([float(c.finish_reason == "length") for _, _, c in flat])
        failed = torch.tensor([g is None for g in graded])
        rewards = torch.tensor([0.0 if g is None else g[0] for g in graded])

        advantages = group_relative_advantages(rewards, len(batch), cfg.group_size)
        # A failed judge call would poison its group's baseline, so the whole group sits out.
        group_failed = failed.view(len(batch), cfg.group_size).any(1, keepdim=True).expand(-1, cfg.group_size)
        advantages = torch.where(group_failed.reshape(-1), 0.0, advantages)
        if cfg.mask_truncated:
            advantages = advantages * (1 - truncated)
        advantages = advantages.cuda()

        adv_list = advantages.tolist()
        live = [k for k, a in enumerate(adv_list) if a != 0.0]
        # With the KL anchor on, masked truncations still get a KL-only pull toward the base model
        # (rambling is where the anchor matters, and an all-truncated step stays recoverable).
        fwd_rows = live + (
            [k for k in range(len(flat)) if adv_list[k] == 0.0 and truncated[k] > 0.5 and not failed[k]]
            if cfg.kl_coef > 0 and cfg.mask_truncated else []
        )
        normalizer = max(1, sum(len(completion_ids[k]) for k in live))
        kl_normalizer = max(1, sum(len(completion_ids[k]) for k in fwd_rows))
        lengths = [len(p) + len(c) for p, c in zip(prompt_ids, completion_ids)]

        optimizer.zero_grad()
        loss_sum = logprob_sum = kl_sum = torch.zeros((), device="cuda")
        for rows in iter_micro_batches(fwd_rows, lengths, cfg.micro_batch_tokens):
            input_ids, attn, comp_mask = build_batch(
                [prompt_ids[k] for k in rows], [completion_ids[k] for k in rows], tokenizer.pad_token_id
            )
            loss, mb_logprobs, mb_kl = grpo_loss(
                model, input_ids, attn, comp_mask, advantages[rows], normalizer, cfg.kl_coef, kl_normalizer
            )
            loss.backward()
            loss_sum, logprob_sum, kl_sum = loss_sum + loss.detach(), logprob_sum + mb_logprobs, kl_sum + mb_kl
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

        t = time.perf_counter()
        lora_request = sync_adapter(model, adapter_dir)
        sync_s = time.perf_counter() - t

        ok = [g for g in graded if g is not None]
        gen_toks = sum(len(c) for c in completion_ids)
        metrics = {
            "train/reward": statistics.mean(g[0] for g in ok) if ok else float("nan"),
            "train/reward_std": statistics.pstdev([g[0] for g in ok]) if ok else float("nan"),
            "train/trunc_rate": truncated.mean().item(),
            "train/mean_completion_len": gen_toks / len(flat),
            "train/token_entropy": -logprob_sum.item() / max(1, sum(len(completion_ids[k]) for k in fwd_rows)),
            "train/zero_adv_frac": 1 - len(live) / len(flat),
            "train/kl": kl_sum.item() / kl_normalizer,
            "train/loss": loss_sum.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "judge/failed": failed.float().mean().item(),
            "judge/latency_s": statistics.median(g[1][f"{reward_judge.name}/latency_s"] for g in ok) if ok else float("nan"),
            "judge/cost": sum(g[1].get(f"{reward_judge.name}/cost", 0.0) for g in ok),
            "perf/gen_s": gen_s,
            "perf/score_s": score_s,
            "perf/sync_s": sync_s,
            "perf/step_s": time.perf_counter() - t_step,
            "perf/gen_tok_per_s": gen_toks / gen_s,
            "perf/peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        log(metrics, step)
        with (out_dir / "rollouts.jsonl").open("a") as f:
            for (task, _, c), g in zip(flat, graded):
                f.write(json.dumps({
                    "step": step, "task_id": task.id, "dimension": task.dimension, "reply": c.text,
                    "truncated": c.finish_reason == "length", "reward": None if g is None else g[0],
                }) + "\n")
        print(
            f"step {step:4d} | reward {metrics['train/reward']:.3f} | trunc {metrics['train/trunc_rate']:.2f} "
            f"| len {metrics['train/mean_completion_len']:.0f} | ent {metrics['train/token_entropy']:.2f} "
            f"| kl {metrics['train/kl']:.4f} | zero_adv {metrics['train/zero_adv_frac']:.2f} "
            f"| gen {gen_s:.0f}s | score {score_s:.0f}s | step {metrics['perf/step_s']:.0f}s "
            f"| peak {metrics['perf/peak_mem_gb']:.1f}GB",
            flush=True,
        )

        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.num_steps:
            # Write to a temp dir and rename into place, so an interrupt can't leave a half-written checkpoint.
            ckpt = Path("checkpoints") / run_name / f"step_{step + 1}"
            tmp = ckpt.with_name(ckpt.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "completed_steps": step + 1}, tmp / "trainer_state.pt")
            shutil.rmtree(ckpt, ignore_errors=True)
            tmp.rename(ckpt)

    if cfg.use_wandb:
        wandb.finish()


# --- launcher ----------------------------------------------------------------------------------


def launch(rewards: list[str], argv: list[str]) -> None:
    """One `train.py --reward <judge>` process per GPU. Engine startup is the PSU's dangerous
    moment (rl-experiments EXPERIMENTS.md §8), so each engine comes up before the next boots."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(rewards) > len(gpus):
        raise SystemExit(f"{len(rewards)} runs but only {len(gpus)} GPUs")
    log_dir = Path("runs") / f"launch-{time.strftime('%m%d-%H%M')}"
    log_dir.mkdir(parents=True)
    procs = {}
    for gpu, reward in zip(gpus, rewards):
        log_path = log_dir / f"{reward}.log"
        with log_path.open("w") as log_file:
            procs[reward] = proc = subprocess.Popen(
                [sys.executable, __file__, *argv, "--reward", reward],
                env=os.environ | {"CUDA_VISIBLE_DEVICES": gpu},
                stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        print(f"{reward}: GPU {gpu}, log {log_path}", flush=True)
        deadline = time.time() + 1800
        while proc.poll() is None and "[ready]" not in log_path.read_text() and time.time() < deadline:
            time.sleep(5)
        time.sleep(30)
    for reward, proc in procs.items():
        print(f"{reward}: exit {proc.wait()}", flush=True)


def parse_args() -> tuple[Config, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rewards", default="", help="comma-separated judge names: one run per GPU, in parallel")
    for f in fields(Config):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=f.default)
    args = parser.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    rewards = [r for r in args.rewards.split(",") if r]
    return cfg, rewards


if __name__ == "__main__":
    cfg, rewards = parse_args()
    if rewards:
        # Pass every other flag through to each run unchanged.
        argv, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
            elif a == "--rewards":
                skip = True
            elif not a.startswith("--rewards="):
                argv.append(a)
        launch(rewards, argv)
    else:
        train(cfg)
