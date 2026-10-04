"""Fine-tune a per-point hit detector with QLoRA.

    python finetune.py prepare --data-dir data
    python finetune.py train --base google/gemma-2-9b-it --out models/hitdet_gemma

`prepare` turns the candidate-claim pairs from `preprocess.py build-pairs` into a balanced
chat-format training set. `train` fine-tunes a 4-bit base model on it with LoRA and writes
the adapter.

The prompt comes from `letitbe.qa_text`, so training and inference see the identical format.
Do not restate the prompt here: one source, or the two drift apart and the detector is
trained on a question it will never be asked.

The adapters released with the paper are classifier artifacts, not the reference coverage
engine. Fine-tuning to a verbatim hit target loosens boilerplate rejection, so the reference
engine stays a zero-shot open model. See the paper for the measurements.

`prepare` also emits an OpenAI chat JSONL, which is the format OpenAI's fine-tuning API
accepts. Uploading it is left to the user; this file does not manage remote jobs.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from letitbe import LABELS, _env, qa_text


def to_chat(rec):
    """A raw pair {cand, claim, aspect, label} -> one chat example. label may be bool or 是/否."""
    label = rec["label"]
    ans = LABELS[label] if isinstance(label, bool) else label
    return {"messages": [
        {"role": "user", "content": qa_text(rec["cand"], rec["claim"], rec["aspect"])},
        {"role": "assistant", "content": ans},
    ]}


def subsample(rows, n):
    """Deterministic balanced subsample: pos : easy_neg : hard_neg = 2 : 1 : 1.

    Within each type, rows are sorted by (aspect, id) and taken at a fixed stride, so every
    aspect is represented and the result does not depend on a random seed.
    """
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    targets = {"pos": n // 2, "easy_neg": n // 4, "hard_neg": n - n // 2 - n // 4}
    out = []
    for t, target in targets.items():
        g = sorted(by_type.get(t, []), key=lambda r: (r["aspect"], r["id"]))
        if not g or target <= 0:
            continue
        step = max(1, len(g) // target)
        out += g[::step][:target]
    return out


def cmd_prepare(args):
    for split, n in (("train", args.train_n), ("val", args.val_n)):
        src = os.path.join(args.data_dir, f"{split}_hits.jsonl")
        if not os.path.exists(src):
            sys.exit(f"{src} not found; run preprocess.py build-pairs first")
        recs = subsample([json.loads(l) for l in open(src)], n)
        out = os.path.join(args.data_dir, f"{split}_sample.openai.jsonl")
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")
        print(f"  {split}: {len(recs)} examples {dict(Counter(r['type'] for r in recs))} "
              f"-> {os.path.basename(out)}")


def cmd_train(args):
    """QLoRA fine-tune, following the Gemma QLoRA cookbook; Mistral-family bases such as
    Breeze use the same PEFT path.

    https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              EarlyStoppingCallback)
    from trl import SFTConfig, SFTTrainer

    train_file = args.train_file or os.path.join(args.data_dir, "train_sample.openai.jsonl")
    val_file = args.val_file or os.path.join(args.data_dir, "val_sample.openai.jsonl")
    token = _env("HF_TOKEN")                         # gated bases such as gemma need this
    print(f"base={args.base}  out={args.out}  train={os.path.basename(train_file)}  "
          f"hf_token={'yes' if token else 'no'}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(args.base, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, dtype=torch.bfloat16,
        device_map="auto", token=token)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset("json", data_files={"train": train_file, "val": val_file})

    def to_text(ex):                                 # {"messages": [...]} -> the base's chat template
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False,
                                                add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(32, len(ds["train"]))))
        ds["val"] = ds["val"].select(range(min(16, len(ds["val"]))))

    cfg = SFTConfig(
        output_dir=args.out,
        # For a base with a large vocabulary (gemma-2-9b has 256k) TRL materialises a
        # (batch*seq, vocab) tensor for per_token_entropy and a batch of 8 runs out of memory.
        # Drop to --batch 4 --grad-accum 4 there; the effective batch is still 16.
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(2 if args.smoke else -1),
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        # Not chunked_nll: TRL's chunked-loss patch hits a num_experts bug on Mistral-family
        # models such as Breeze. Plain nll is mathematically identical here.
        loss_type="nll",
        # Breeze is a dense Mistral with no mixture-of-experts router, but MistralConfig carries
        # output_router_logits, which makes TRL read it as MoE and reach for an outputs.aux_loss
        # that never exists. Setting this to 0 disables that path and is correct for a dense model.
        router_aux_loss_coef=0.0,
        eval_strategy=("no" if args.smoke else "epoch"),
        save_strategy=("no" if args.smoke else "epoch"),
        load_best_model_at_end=(not args.smoke),     # keep the epoch with the lowest validation loss
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, bf16=True, max_length=2048, packing=False,
        report_to="none", dataset_text_field="text")

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds["train"],
        eval_dataset=(None if args.smoke else ds["val"]),
        peft_config=lora, processing_class=tok,
        callbacks=([] if args.smoke
                   else [EarlyStoppingCallback(early_stopping_patience=args.patience)]))
    trainer.train()
    if args.smoke:
        print("smoke ok: the configuration runs. No adapter was saved.")
        return
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"done -> {args.out}")


def selfcheck():
    """Subsample balance and determinism, and the chat example's shape."""
    rows = [{"id": f"T{i:04d}", "cand": "c", "claim": "k", "aspect": ["外觀", "差異"][i % 2],
             "label": i % 4 == 0, "type": ["pos", "pos", "easy_neg", "hard_neg"][i % 4]}
            for i in range(400)]
    s = subsample(rows, 40)
    c = Counter(r["type"] for r in s)
    assert c["pos"] == 20 and c["easy_neg"] == 10 and c["hard_neg"] == 10, c
    assert subsample(rows, 40) == s, "subsample must not depend on a random seed"
    assert len({r["aspect"] for r in s}) == 2, "every aspect should survive the stride"

    ex = to_chat({"cand": "二者都有鍋寶二字", "claim": "兩商標共同含『鍋寶』",
                  "aspect": "主要識別部分", "label": True})
    assert ex["messages"][0]["content"] == qa_text("二者都有鍋寶二字", "兩商標共同含『鍋寶』",
                                                   "主要識別部分"), "prompt must come from letitbe"
    assert ex["messages"][1]["content"] == "是"
    assert to_chat({"cand": "x", "claim": "y", "aspect": "外觀",
                    "label": False})["messages"][1]["content"] == "否"
    print("selfcheck ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="balanced chat-format training set from the raw pairs")
    p.add_argument("--train-n", type=int, default=3000)
    p.add_argument("--val-n", type=int, default=500)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="QLoRA fine-tune a 4-bit base model")
    p.add_argument("--base", required=True, help="Hugging Face repo of the base model")
    p.add_argument("--out", required=True, help="where to write the adapter")
    p.add_argument("--train-file", help="override the prepared training file")
    p.add_argument("--val-file", help="override the prepared validation file")
    p.add_argument("--batch", type=int, default=8, help="per-device micro-batch")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10, help="upper bound; early stopping decides")
    p.add_argument("--patience", type=int, default=2, help="epochs without eval_loss improvement")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smoke", action="store_true", help="2 steps on 32 examples, saves nothing")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("selfcheck", help="run the internal consistency checks and exit")
    p.set_defaults(func=lambda args: selfcheck())

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
