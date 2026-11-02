"""DPO of the SFT gauge model on the pairs from sample_pairs.py.

    python vlm/dpo.py /data/datasets/gauges_vlm /data/runs/vlm/dpo/pairs.jsonl /data/runs/vlm/dpo/b0.1 \\
        --adapter /data/runs/vlm/r8/adapter

Run in a pod (see submit.sh). The 4-bit language model stays frozen. Training continues the SFT LoRA adapter.
The reference model is the SFT adapter: its log-probabilities of each chosen and rejected answer are
calculated once before training and saved to <out>/ref_logps.json.

Loss (Rafailov et al., 2023), with log-probabilities summed over the answer tokens:
    -log sigmoid(beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)))
The prompt and answer format is the same as in train.py: prompt + answer + <|im_end|>, loss only on the answer.
No augmentation, so the policy and the reference see the same image. Dropout is off, so at step 0 the
policy equals the reference and the loss is log 2 = 0.693.
Logs to W&B if WANDB_API_KEY is set. Resumes from the last checkpoint and writes a DONE marker on completion.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL, load, read_jsonl  # noqa: E402
from train import Collator  # noqa: E402


class PairCollator(Collator):
    """Rows 0..n-1 of the batch are the chosen answers, rows n..2n-1 the rejected answers, for the same images."""

    def __call__(self, pairs):
        rows = [{"image": p["image"], "answer": p["chosen"]} for p in pairs] + \
               [{"image": p["image"], "answer": p["rejected"]} for p in pairs]
        batch = super().__call__(rows)
        if "ref_chosen" in pairs[0]:
            batch["ref_chosen"] = torch.tensor([p["ref_chosen"] for p in pairs])
            batch["ref_rejected"] = torch.tensor([p["ref_rejected"] for p in pairs])
        return batch


def answer_logps(model, batch):
    """Sum of the log-probabilities of the answer tokens (labels != -100), one value per row."""
    labels = batch.pop("labels")[:, 1:]
    logits = model(**batch).logits[:, :-1]
    mask = labels != -100
    tok = logits[mask].float().log_softmax(-1).gather(-1, labels[mask].unsqueeze(-1)).squeeze(-1)
    return torch.zeros(labels.shape[0], device=tok.device).index_add(0, mask.nonzero()[:, 0], tok)


@torch.no_grad()
def reference_logps(model, collator, pairs, batch_size):
    model.eval()
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        lp = answer_logps(model, {k: v.to(model.device) for k, v in collator(chunk).items()}).cpu()
        for n, p in enumerate(chunk):
            p["ref_chosen"], p["ref_rejected"] = float(lp[n]), float(lp[n + len(chunk)])
        if (i // batch_size) % 50 == 0:
            print(f"reference log-probs {i + len(chunk)}/{len(pairs)}", flush=True)


class DPOTrainer(Trainer):
    loss_is_scaled_for_ga = False  # compute_loss returns a mean over pairs; Trainer divides it by the accumulation steps

    def __init__(self, *args, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.stats = {"train": defaultdict(list), "eval": defaultdict(list)}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ref_c, ref_r = inputs.pop("ref_chosen"), inputs.pop("ref_rejected")
        lp = answer_logps(model, inputs)
        n = ref_c.shape[0]
        rc, rr = self.beta * (lp[:n] - ref_c), self.beta * (lp[n:] - ref_r)  # implicit rewards
        loss = -F.logsigmoid(rc - rr).mean()
        s = self.stats["train" if model.training else "eval"]
        for k, v in {"rewards/chosen": rc, "rewards/rejected": rr, "rewards/margin": rc - rr,
                     "rewards/accuracy": (rc > rr).float(), "logps/chosen": lp[:n], "logps/rejected": lp[n:]}.items():
            s[k].append(v.detach().mean().item())
        return (loss, {}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        mode = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        prefix = "eval_" if mode == "eval" else ""
        logs.update({prefix + k: sum(v) / len(v) for k, v in self.stats[mode].items() if v})
        self.stats[mode].clear()
        super().log(logs, *args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--adapter", required=True, help="the SFT adapter: start point and reference")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=2, help="pairs per device step (2 x batch sequences)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--val-n", type=int, default=100, help="pairs held out for the eval loss and accuracy")
    ap.add_argument("--limit", type=int, help="only the first N pairs, for a quick check")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "DONE").exists():
        return print(f"{out} already finished")
    out.mkdir(parents=True, exist_ok=True)

    from peft import PeftModel
    model, processor = load(args.model)
    processor.tokenizer.padding_side = "right"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.print_trainable_parameters()

    collator = PairCollator(processor, args.data)
    ref_file = out / "ref_logps.json"
    if ref_file.exists():
        pairs = json.loads(ref_file.read_text())
    else:
        pairs = read_jsonl(args.pairs)[:args.limit]
        random.Random(0).shuffle(pairs)
        reference_logps(model, collator, pairs, args.batch)  # same batch size as training
        ref_file.write_text(json.dumps(pairs))
    val_n = min(args.val_n, len(pairs) // 10)
    val, train = pairs[:val_n], pairs[val_n:]
    print(f"{len(train)} training pairs, {len(val)} eval pairs")

    steps = max(len(train) // (args.batch * args.accum), 1)
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, lr_scheduler_type="cosine", warmup_steps=0.1,
        bf16=torch.cuda.is_available(), optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=1 if args.limit else 5, eval_strategy="steps", eval_steps=max(steps // 4, 1),
        save_strategy="steps", save_steps=max(steps // 4, 1), save_total_limit=2, remove_unused_columns=False,
        label_names=["labels"], dataloader_num_workers=2, prediction_loss_only=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none", run_name=f"dpo-{out.name}")

    trainer = DPOTrainer(model=model, args=targs, train_dataset=train, eval_dataset=val, data_collator=collator,
                         beta=args.beta)
    resume = any(out.glob("checkpoint-*"))
    if not resume:
        print("step 0:", trainer.evaluate())  # expect eval_loss 0.693 and rewards 0
    trainer.train(resume_from_checkpoint=True if resume else None)
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
