import math
import os
from datetime import datetime
from contextlib import nullcontext
import time
from functools import partial

import torch

from model import ModelArgs, Transformer
from tinystories import Task


#------------------- I/O -------------------#
out_dir = "out"
eval_interval = 10
log_interval = 1
eval_iters = 10
eval_only = False
always_save_checkpoint = False
init_from = "scratch"
# wandb logging
wandb_log = False
wandb_project = "saturnllama"
wandb_run_name = "run" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#data
batch_size = 32
max_seq_len = 256
vocab_source = "llama2"
vocab_size = 32000
# model
dim = 32
n_layers = 3
n_heads = 3
multiple_of = 3
dropout = 0.0
#adamw optimizer
gradient_accumulation_steps = 4
learning_rate = 5e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay
decay_lr = True
warmup_iters = 10
# system
device = "mps"  # macbook
dtype = "bfloat16"
compile = False

# put above configs into a dict
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, str, bool))
]
config = {k: globals()[k] for k in config_keys}

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters
min_lr = 0.0

assert vocab_source in ["llama2", "custom"]
assert vocab_size == 32000

# single gpu and one process
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iter: {tokens_per_iter}")
    print(f"tokens per epoch: {tokens_per_iter * max_iters}")
    print(f"tokens per epoch: {tokens_per_iter * max_iters / 1e9} Billion")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1984 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.alow_tf32 = True
device_type = "cuda" if device.startswith("cuda") else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.cuda.amp.autocast(device_type=device_type, detype=ptdtype)
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# init these up here, can override if init_from = "resume"
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)
if init_from == "scratch":
    print("Initializing model from scratch")
    gptconfig = ModelArgs(**model_args)
    model = Transformer(gptconfig)
elif init_from == "resume":
    raise NotImplementedError
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# if init_from == "resume" and "optimizer" in checkpoint:
#     optimizer.load_state_dict(checkpoint["optimizer"])

# compile the model
if compile:
    print("Compiling model")
    unoptimized_model = model
    model = torch.compile(model)

# helps estimate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # disable dropout etc.
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # re-enable dropout etc.
    return out

# learning rate decay scheduler (cosine decay with warmup)
def get_lr(it):
    if it < warmup_iters:  # linear warmup
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + (learning_rate - min_lr) * coeff

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t0 = time.time()
local_iter_num = 0
raw_model = model
running_mfu = -1.0
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }, step = iter_num,
                )
            except Exception as e:
                print(f"wandb logging failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"Saving checkpoint to {out_dir}/checkpoint.pt")
                torch.save(checkpoint, f"{out_dir}/checkpoint.pt")
    if iter_num == 0 and eval_only:
        break

    # forward pass update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(
            f"step {iter_num} | loss: {lossf:.4f} | lr: {lr:.4f} | dt: {dt:.4f} | tokens/sec: {tokens_per_iter / dt:.1f}"
        )
    iter_num += 1
    local_iter_num += 1

    # terminate after max_iters
    if iter_num >= max_iters:
        break