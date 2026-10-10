"""QLoRA fine-tune of Qwen3-VL on rectified gauge images.

    python vlm/train.py /data/datasets/gauges_vlm /data/runs/vlm/r8 --epochs 2

Run in a pod (see submit.sh). The 4-bit language model is frozen. Only LoRA adapters on the attention
and MLP projections are trained. The loss is masked to the answer tokens. Logs to W&B if WANDB_API_KEY is set.
Resumes from the last checkpoint after a pod restart, and writes a DONE marker on completion.
"""
import argparse
import os
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL, load, messages, read_jsonl  # noqa: E402

# Rotating the whole dial rotates the scale with the needle, so the reading is unchanged.
AUGMENT = transforms.Compose([
    transforms.RandomRotation(10, fill=(127, 127, 127)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.02),
])


class Collator:
    def __init__(self, processor, root):
        self.processor, self.root = processor, Path(root)
        self.prompt = processor.apply_chat_template(messages(), add_generation_prompt=True, tokenize=False)
        self.end = "<|im_end|>"

    def image(self, row):
        im = Image.open(self.root / row["image"]).convert("RGB")
        if row.get("augment"):
            im = AUGMENT(im)
            if random.random() < 0.2:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
        return im

    def __call__(self, rows):
        texts = [self.prompt + r["answer"] + self.end for r in rows]
        batch = self.processor(text=texts, images=[self.image(r) for r in rows], padding=True, return_tensors="pt")
        # Mask all but the last k non-padding tokens (the answer) of each row.
        labels = torch.full_like(batch["input_ids"], -100)
        lengths = batch["attention_mask"].sum(1)
        for i, r in enumerate(rows):
            k = len(self.processor.tokenizer(r["answer"] + self.end, add_special_tokens=False)["input_ids"])
            n = int(lengths[i])
            labels[i, n - k:n] = batch["input_ids"][i, n - k:n]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N training images, for a quick check")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "DONE").exists():
        return print(f"{out} already finished")

    model, processor = load(args.model)
    processor.tokenizer.padding_side = "right"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model.print_trainable_parameters()

    train = [r | {"augment": True} for r in read_jsonl(Path(args.data) / "train.jsonl")[:args.limit]]
    val = read_jsonl(Path(args.data) / "val.jsonl")[:16 if args.limit else None]  # no augmentation
    steps = max(len(train) // (args.batch * args.accum), 1)
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, optim="paged_adamw_8bit", logging_steps=5,
        eval_strategy="steps", eval_steps=max(steps // 4, 1), save_strategy="steps", save_steps=max(steps // 4, 1),
        save_total_limit=2, remove_unused_columns=False, dataloader_num_workers=2,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none", run_name=out.name)

    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(processor, args.data))
    resume = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
