"""GRPO with LoRA on HumanAgencyBench, with one reward judge per run and rollouts on vLLM.

Ported from rl-experiments/train.py: vLLM samples each GRPO group, a PEFT LoRA trainer updates the
policy, and each step's adapter reaches vLLM through tmpfs and an in-place LoRARequest reload
(~0.5 s, no base-weight copy). vLLM and the trainer share one GPU: the engine sleeps (weights
offloaded to CPU, KV cache freed) while the trainer runs, so a GPU hosts a whole run. A run can
also span several GPUs: one process per GPU, each with its own engine and trainer replica taking
a share of every step's prompts, with LoRA gradients summed across processes.

Defaults follow Prime Intellect's prime-rl examples for Qwen3-4B-Instruct-2507 (LoRA, lr 1e-5,
16 prompts x 16 rollouts, 200 steps). Every `eval_every` steps, starting at step 0, held-out
prompts are answered greedily and graded by the whole judges.json panel.

    uv run --env-file .env train.py --reward jev_latest                     # one run, one GPU
    uv run --env-file .env train.py --reward jev_latest --gpus-per-run 3    # one run on 3 GPUs
    uv run --env-file .env train.py --rewards jev_latest,mercury_3          # one run per GPU
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
    min_group_spread: float = 0.02
    """Groups whose rewards span less than this sit out: their (unscaled) advantages carry almost no
    gradient but still cost a full forward and backward."""
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
    judge_concurrency: int = 64

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
    sleep so the trainer has the GPU. Returns the outputs and the seconds spent waking and sleeping."""
    import vllm

    t = time.perf_counter()
    # The trainer's freed activations stay cached by PyTorch; hand them back before vLLM
    # reclaims its weights and KV cache, or the wake-up runs out of memory.
    torch.cuda.empty_cache()
    llm.wake_up()
    handoff_s = time.perf_counter() - t
    try:
        outs = llm.chat(
            [t.messages for t in tasks],
            sampling_params=vllm.SamplingParams(n=n, temperature=temperature, max_tokens=cfg.max_new_tokens),
            lora_request=lora_request,
            use_tqdm=False,
        )
    finally:
        t = time.perf_counter()
        llm.sleep(level=1)
        handoff_s += time.perf_counter() - t
    return outs, handoff_s


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


def group_relative_advantages(rewards, num_prompts, group_size, min_spread: float):
    """r - mean within each prompt's group, as prime-rl does ("Dr. GRPO": no division by the group's
    std, which would blow a decision judge's near-tied continuous scores up into full-size advantages
    while a chat judge's whole-point scores tie exactly). Groups spanning less than min_spread get 0."""
    grouped = rewards.view(num_prompts, group_size)
    advantages = grouped - grouped.mean(1, keepdim=True)
    spread = grouped.max(1, keepdim=True).values - grouped.min(1, keepdim=True).values
    return torch.where(spread < max(min_spread, 1e-9), 0.0, advantages).reshape(-1)


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


# --- distributed -------------------------------------------------------------------------------

# Set by `launch` for a run spread over several GPUs. Only LoRA gradients and small stats cross
# processes, over gloo on the CPU (~16M parameters: a fraction of a second per step), which keeps
# torch.distributed clear of the GPUs vLLM manages.
RANK = int(os.environ.get("JEV_RANK", "0"))
WORLD = int(os.environ.get("JEV_WORLD_SIZE", "1"))


def init_dist() -> None:
    if WORLD > 1:
        import torch.distributed as dist

        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{os.environ['JEV_DIST_PORT']}", rank=RANK, world_size=WORLD
        )


def broadcast_object(obj):
    if WORLD == 1:
        return obj
    import torch.distributed as dist

    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def gather_objects(obj) -> list:
    if WORLD == 1:
        return [obj]
    import torch.distributed as dist

    out = [None] * WORLD
    dist.all_gather_object(out, obj)
    return out


def sum_across_ranks(*values: float) -> list[float]:
    if WORLD == 1:
        return list(values)
    import torch.distributed as dist

    t = torch.tensor(values, dtype=torch.float64)
    dist.all_reduce(t)
    return t.tolist()


def sync_lora(params, grads: bool) -> None:
    """Sum the trainable parameters' gradients across ranks (grads=True), or copy rank 0's
    parameters to every rank (grads=False, once at start, so the replicas begin identical)."""
    if WORLD == 1:
        return
    import torch.distributed as dist

    if grads:
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float().cpu() for p in params])
        dist.all_reduce(flat)
    else:
        flat = torch.cat([p.detach().reshape(-1).float().cpu() for p in params])
        dist.broadcast(flat, src=0)
    offset = 0
    for p in params:
        chunk = flat[offset : offset + p.numel()].view(p.shape).to(p.device, p.dtype)
        if grads:
            p.grad = chunk
        else:
            p.data.copy_(chunk)
        offset += p.numel()


# --- eval ----------------------------------------------------------------------------------------


def eval_tasks(cfg: Config) -> list[Task]:
    test = load_tasks("test")
    random.Random(cfg.seed).shuffle(test)
    per_dim = cfg.eval_n // len(RUBRICS)
    return [t for dim in sorted(RUBRICS) for t in [t for t in test if t.dimension == dim][:per_dim]]


def evaluate(step, llm, panel, tasks, cfg, lora_request, loop, out_dir: Path) -> dict | None:
    """Greedy replies to held-out prompts (each rank answers its share), graded by every judge.
    Rank 0 saves each reply with all judge scores to eval_step_<step>.jsonl and returns eval/*
    metrics; other ranks return None."""
    t = time.perf_counter()
    shard = tasks[RANK::WORLD]
    outs, _ = generate(llm, shard, 1, 0.0, cfg, lora_request)
    replies = [o.outputs[0].text for o in outs]

    async def grade_panel():  # gather inside a coroutine, so its future belongs to `loop`
        return await asyncio.gather(*(
            grade_all(judge, list(zip(shard, replies)), cfg.judge_concurrency) for judge in panel
        ))

    results = loop.run_until_complete(grade_panel())
    records = [
        {
            "task_id": task.id, "dimension": task.dimension, "reply": reply,
            "truncated": out.outputs[0].finish_reason == "length", "tokens": len(out.outputs[0].token_ids),
            "judges": {j.name: (None if res[i] is None else {"score": res[i][0], **res[i][1]})
                       for j, res in zip(panel, results)},
        }
        for i, (task, reply, out) in enumerate(zip(shard, replies, outs))
    ]
    records = [r for rank_records in gather_objects(records) for r in rank_records]
    if RANK != 0:
        return None
    with (out_dir / f"eval_step_{step}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    metrics = {
        "eval/trunc_rate": statistics.mean(r["truncated"] for r in records),
        "eval/mean_len": statistics.mean(r["tokens"] for r in records),
    }
    for judge in panel:
        ok = [r["judges"][judge.name] for r in records if r["judges"][judge.name] is not None]
        metrics[f"eval/{judge.name}/failed"] = 1 - len(ok) / len(records)
        if ok:
            metrics[f"eval/{judge.name}"] = statistics.mean(o["score"] for o in ok)
            metrics[f"eval/{judge.name}/latency_s"] = statistics.median(o[f"{judge.name}/latency_s"] for o in ok)
    # Every policy looks good to the judge it trained on; compare runs on the rest of the panel.
    heldout = [metrics[f"eval/{j.name}"] for j in panel if not j.weight and f"eval/{j.name}" in metrics]
    if heldout:
        metrics["eval/heldout_mean"] = statistics.mean(heldout)
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
    train_tasks = load_tasks("train")
    held_out = eval_tasks(cfg)
    print(f"[data] {len(train_tasks)} train prompts, {len(held_out)} eval prompts; reward = {reward_judge.name}", flush=True)

    # Build the engine first, while the GPU is empty, and before torch.distributed starts, so
    # vLLM's engine process never sees our process group; then sleep it so the trainer can load.
    llm = build_engine(cfg)
    llm.sleep(level=1)
    print("[ready] vLLM engine up", flush=True)
    init_dist()

    run_name = out_dir = None
    if RANK == 0:
        run_name = (
            f"{reward_judge.name}-{cfg.model.split('/')[-1].lower()}-r{cfg.lora_rank}-lr{cfg.lr:g}"
            f"{f'-x{WORLD}' if WORLD > 1 else ''}-{time.strftime('%m%d-%H%M')}"
        )
        (Path("runs") / run_name).mkdir(parents=True)
        (Path("runs") / run_name / "config.json").write_text(
            json.dumps(asdict(cfg) | {"run_name": run_name, "gpus": WORLD}, indent=2)
        )
    run_name = broadcast_object(run_name)  # also holds the other ranks until rank 0 made out_dir
    out_dir = Path("runs") / run_name

    model, tokenizer = load_policy(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    sync_lora(trainable, grads=False)
    optimizer = bnb.optim.AdamW8bit(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps)
    adapter_dir = f"/dev/shm/jev-as-a-judge/{run_name}/rank{RANK}"
    lora_request = sync_adapter(model, adapter_dir)
    # One loop for the whole run: the judges' HTTP clients are bound to the loop they first ran on.
    loop = asyncio.new_event_loop()

    if cfg.use_wandb and RANK == 0:
        import wandb

        wandb.init(project=cfg.wandb_project, name=run_name, config=asdict(cfg) | {"gpus": WORLD})

    def log(metrics: dict, step: int) -> None:
        if cfg.use_wandb:
            wandb.log(metrics, step=step)
        with (out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")

    # Every rank draws the same batch and takes its share of the prompts.
    prompt_batches = iter_prompt_batches(train_tasks, cfg.prompts_per_step, cfg.seed)
    for step in range(cfg.num_steps + 1):
        if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.num_steps):
            eval_metrics = evaluate(step, llm, panel, held_out, cfg, lora_request, loop, out_dir)
            if RANK == 0:
                log(eval_metrics, step)
        if step == cfg.num_steps:
            break

        t_step = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        batch = next(prompt_batches)[RANK::WORLD]
        t = time.perf_counter()
        outs, handoff_s = generate(llm, batch, cfg.group_size, cfg.temperature, cfg, lora_request)
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

        advantages = group_relative_advantages(rewards, len(batch), cfg.group_size, cfg.min_group_spread)
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
        # Normalize by the whole step's token counts, across ranks, so the summed gradients equal
        # what one GPU would have computed for the full batch.
        trained_toks, fwd_toks, global_fwd_rows = sum_across_ranks(
            sum(len(completion_ids[k]) for k in live), sum(len(completion_ids[k]) for k in fwd_rows), len(fwd_rows)
        )
        normalizer, kl_normalizer = max(1, trained_toks), max(1, fwd_toks)
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
        grad_norm = 0.0
        if global_fwd_rows:  # an all-tied step has no gradient anywhere; don't let Adam momentum move it
            sync_lora(trainable, grads=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm).item()
            optimizer.step()
        scheduler.step()

        t = time.perf_counter()
        lora_request = sync_adapter(model, adapter_dir)
        sync_s = time.perf_counter() - t

        with (out_dir / f"rollouts_rank{RANK}.jsonl").open("a") as f:
            for (task, _, c), g in zip(flat, graded):
                f.write(json.dumps({
                    "step": step, "task_id": task.id, "dimension": task.dimension, "reply": c.text,
                    "truncated": c.finish_reason == "length", "reward": None if g is None else g[0],
                }) + "\n")
        ranks = gather_objects({
            "rewards": [g[0] for g in graded if g is not None],
            "latency": [g[1][f"{reward_judge.name}/latency_s"] for g in graded if g is not None],
            "cost": sum(g[1].get(f"{reward_judge.name}/cost", 0.0) for g in graded if g is not None),
            "n": len(flat), "failed": int(failed.sum()), "truncated": float(truncated.sum()),
            "gen_toks": sum(len(c) for c in completion_ids), "live": len(live),
            "logprob_sum": logprob_sum.item(), "kl_sum": kl_sum.item(), "loss": loss_sum.item(),
            "gen_s": gen_s, "handoff_s": handoff_s, "score_s": score_s, "sync_s": sync_s,
            "peak": torch.cuda.max_memory_allocated() / 1e9,
        })

        if RANK == 0:
            rew = [r for s in ranks for r in s["rewards"]]
            lat = [x for s in ranks for x in s["latency"]]
            n = sum(s["n"] for s in ranks)
            gen_toks = sum(s["gen_toks"] for s in ranks)
            slowest = lambda key: max(s[key] for s in ranks)
            metrics = {
                "train/reward": statistics.mean(rew) if rew else float("nan"),
                "train/reward_std": statistics.pstdev(rew) if rew else float("nan"),
                "train/trunc_rate": sum(s["truncated"] for s in ranks) / n,
                "train/mean_completion_len": gen_toks / n,
                "train/token_entropy": -sum(s["logprob_sum"] for s in ranks) / kl_normalizer,
                "train/zero_adv_frac": 1 - sum(s["live"] for s in ranks) / n,
                "train/kl": sum(s["kl_sum"] for s in ranks) / kl_normalizer,
                "train/loss": sum(s["loss"] for s in ranks),
                "train/grad_norm": grad_norm,
                "train/lr": scheduler.get_last_lr()[0],
                "judge/failed": sum(s["failed"] for s in ranks) / n,
                "judge/latency_s": statistics.median(lat) if lat else float("nan"),
                "judge/cost": sum(s["cost"] for s in ranks),
                "perf/gen_s": slowest("gen_s"),
                "perf/handoff_s": slowest("handoff_s"),
                "perf/score_s": slowest("score_s"),
                "perf/sync_s": slowest("sync_s"),
                "perf/step_s": time.perf_counter() - t_step,
                "perf/gen_tok_per_s": gen_toks / slowest("gen_s"),
                "perf/peak_mem_gb": slowest("peak"),
            }
            log(metrics, step)
            print(
                f"step {step:4d} | reward {metrics['train/reward']:.3f} | trunc {metrics['train/trunc_rate']:.2f} "
                f"| len {metrics['train/mean_completion_len']:.0f} | ent {metrics['train/token_entropy']:.2f} "
                f"| kl {metrics['train/kl']:.4f} | zero_adv {metrics['train/zero_adv_frac']:.2f} "
                f"| gen {metrics['perf/gen_s']:.0f}s | score {metrics['perf/score_s']:.0f}s "
                f"| step {metrics['perf/step_s']:.0f}s | peak {metrics['perf/peak_mem_gb']:.1f}GB",
                flush=True,
            )

        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.num_steps:
            # The replicas apply identical updates; check they haven't drifted apart.
            sums = gather_objects(sum(p.detach().float().sum().item() for p in trainable))
            if max(sums) - min(sums) > 1e-3 * max(1.0, abs(sums[0])):
                print(f"  [warn] LoRA replicas differ across ranks: {sums}", flush=True)
            if RANK == 0:
                # Write to a temp dir and rename into place, so an interrupt can't leave a half-written checkpoint.
                ckpt = Path("checkpoints") / run_name / f"step_{step + 1}"
                tmp = ckpt.with_name(ckpt.name + ".tmp")
                shutil.rmtree(tmp, ignore_errors=True)
                model.save_pretrained(tmp)
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "completed_steps": step + 1}, tmp / "trainer_state.pt")
                shutil.rmtree(ckpt, ignore_errors=True)
                tmp.rename(ckpt)

    # Exiting with the engine asleep makes vLLM's teardown fail on its offloaded memory, which
    # would mark a finished run as failed.
    torch.cuda.empty_cache()
    llm.wake_up()
    if cfg.use_wandb and RANK == 0:
        wandb.finish()
    if WORLD > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


# --- launcher ----------------------------------------------------------------------------------


def launch(rewards: list[str], gpus_per_run: int, argv: list[str]) -> None:
    """Runs `train.py --reward <judge>` for each reward judge on its own `gpus_per_run` GPUs, one
    process per GPU. Engine startup is the PSU's dangerous moment (rl-experiments EXPERIMENTS.md
    §8), so each process's engine comes up before the next one boots."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(rewards) * gpus_per_run > len(gpus):
        raise SystemExit(f"{len(rewards)} runs x {gpus_per_run} GPUs each, but only {len(gpus)} GPUs")
    log_dir = Path("runs") / f"launch-{time.strftime('%m%d-%H%M')}"
    log_dir.mkdir(parents=True)
    procs = {}
    for j, reward in enumerate(rewards):
        for rank in range(gpus_per_run):
            gpu = gpus[j * gpus_per_run + rank]
            name = reward if gpus_per_run == 1 else f"{reward}-rank{rank}"
            log_path = log_dir / f"{name}.log"
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": gpu, "JEV_RANK": str(rank),
                "JEV_WORLD_SIZE": str(gpus_per_run), "JEV_DIST_PORT": str(29500 + j),
            }
            with log_path.open("w") as log_file:
                procs[name] = proc = subprocess.Popen(
                    [sys.executable, __file__, *argv, "--reward", reward],
                    env=env, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                )
            print(f"{name}: GPU {gpu}, log {log_path}", flush=True)
            deadline = time.time() + 1800
            while proc.poll() is None and "[ready]" not in log_path.read_text() and time.time() < deadline:
                time.sleep(5)
            time.sleep(30)
    for name, proc in procs.items():
        print(f"{name}: exit {proc.wait()}", flush=True)


LAUNCH_FLAGS = ("--rewards", "--gpus-per-run")


def parse_args() -> tuple[Config, list[str], int]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rewards", default="", help="comma-separated judge names: one run each, in parallel")
    parser.add_argument("--gpus-per-run", type=int, default=1, help="GPUs per run (one process each)")
    for f in fields(Config):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=f.default)
    args = parser.parse_args()
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
    rewards = [r for r in args.rewards.split(",") if r]
    return cfg, rewards, args.gpus_per_run


if __name__ == "__main__":
    cfg, rewards, gpus_per_run = parse_args()
    if "JEV_RANK" not in os.environ and (rewards or gpus_per_run > 1):
        # Pass every other flag through to each process unchanged.
        argv, skip = [], False
        for a in sys.argv[1:]:
            if skip:
                skip = False
            elif a in LAUNCH_FLAGS:
                skip = True
            elif not a.startswith(tuple(f + "=" for f in LAUNCH_FLAGS)):
                argv.append(a)
        launch(rewards or [cfg.reward], gpus_per_run, argv)
    else:
        train(cfg)
