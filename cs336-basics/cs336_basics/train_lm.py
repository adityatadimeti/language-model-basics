import sys
import os
import time
import math
import yaml
import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.model_utils import load_data, save_checkpoint, load_checkpoint, get_cluster_data_path
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_modules import TransformerLM, cross_entropy, perplexity, decode
from cs336_basics.optimizer import (
    learning_rate_schedule,
    gradient_clipping,
    SGD,
    AdamW,
)




def evaluate(
    model: torch.nn.Module,
    val_data: np.memmap,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_batches: int = 50,
) -> float:
    """
    Draw `eval_batches` random batches from val_data, compute avg cross-entropy loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            xb, yb = load_data(val_data, batch_size, context_length, device)
            logits = model(xb)
            B, T, V = logits.shape
            loss = cross_entropy(
                logits.view(-1, V),           
                yb.reshape(-1)                
            )
            
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_lm(cfg):
    # Initialize Weights & Biases run
    experiment_name = cfg.get('experiment_name', None)
    wandb.init(
        project=cfg.get('wandb_project', 'transformer_lm'),
        config=cfg,
        name=experiment_name,
        resume=cfg.get('resume_id', None)
    )
    experiment_name = wandb.run.name

    run_ckpt_path = None
    # Check if checkpoint_directory is specified in the config
    if 'checkpoint_directory' in cfg and cfg['checkpoint_directory']:
        run_ckpt_path = os.path.join(cfg['checkpoint_directory'], experiment_name)
        os.makedirs(run_ckpt_path, exist_ok=True)
        print(f"Saving checkpoints to: {run_ckpt_path}")
    else:
        print("No checkpoint directory specified in config, checkpoints will not be saved")





    # Hyperparameters   
    vocab_size        = int(cfg['vocab_size'])
    context_length    = int(cfg.get('context_length', 128))
    d_model           = int(cfg.get('d_model', 512))
    d_ff              = int(cfg.get('d_ff', 2048))
    num_heads         = int(cfg.get('num_heads', 8))
    num_layers        = int(cfg.get('num_layers', 6))
    batch_size        = int(cfg.get('batch_size', 32))
    max_iters         = int(cfg.get('max_iters', 10000))
    max_lr            = float(cfg.get('max_lr', 1e-3))
    min_lr            = float(cfg.get('min_lr', 1e-5))
    warmup_iters_frac = float(cfg.get('warmup_iters_frac', 0.05))
    cosine_cycle_frac = float(cfg.get('cosine_cycle_frac', 0.8))
    assert 0.0 <= warmup_iters_frac <= 1.0, "warmup_frac must be in [0,1]"
    assert 0.0 <= cosine_cycle_frac <= 1.0, "cosine_frac must be in [0,1]"

    warmup_iters       = int(warmup_iters_frac * max_iters)
    cosine_cycle_iters = min(int(cosine_cycle_frac * max_iters), max_iters)

    weight_decay      = float(cfg.get('weight_decay', 1e-2))
    beta1             = float(cfg.get('beta1', 0.9))
    beta2             = float(cfg.get('beta2', 0.999))
    eps               = float(cfg.get('eps', 1e-8))
    max_grad_norm     = float(cfg.get('max_grad_norm', 1.0))
    optimizer_type    = str(cfg.get('optimizer', 'adamw')).lower()
    log_interval      = int(cfg.get('log_interval', 100))
    val_interval      = int(cfg.get('val_interval', 500))
    save_interval     = int(cfg.get('save_interval', 1000))
    theta             = int(cfg.get('theta', 0))
    gradient_accum    = int(cfg.get('gradient_accumulation_steps', 1))
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    print(f"Training on device: {device}")


    train_data_path = get_cluster_data_path(cfg['train_data'])
    val_data_path   = get_cluster_data_path(cfg['val_data'])
    full_train = np.load(train_data_path, mmap_mode='r')
    full_val   = np.load(val_data_path,   mmap_mode='r')

    if 'max_data_tokens' in cfg:
        N = int(cfg['max_data_tokens'])
        # +1 so we can form a context→target pair
        full_train = full_train[: N + context_length + 1]
        full_val   = full_val[:   N + context_length + 1]
    # train_data = full_train.astype(np.int64)
    # val_data   = full_val.astype(np.int64)

    train_data = full_train
    val_data = full_val




    resume          = cfg.get('resume', None)


    # Model
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        weights={},
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        theta=theta if theta else None,
        token_positions=None,
        max_seq_len=context_length
    ).to(device)

    model = torch.compile(model)

    # Optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=max_lr)
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )

    # Resume checkpoint
    start_iter = 0
    if resume:
        start_iter = load_checkpoint(resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    it = start_iter
    start_time = time.time()

    optimizer.zero_grad()

    # Training loop
    with tqdm(total=max_iters) as pbar:
        while it < max_iters:

            xb, yb = load_data(train_data, batch_size, context_length, device)
            logits = model(xb)
            B, T, V = logits.shape

            # LR schedule
            lr = learning_rate_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
            for group in optimizer.param_groups:
                group['lr'] = lr

            # Loss and update
            # loss = cross_entropy(logits.view(B*T, V), yb.view(B*T))
            loss = cross_entropy(
                logits.view(-1, V),          
                yb.reshape(-1)               
            )
            loss = loss / gradient_accum
            loss.backward()


            if max_grad_norm > 0:
                gradient_clipping(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            
            it += 1
            pbar.update(1)

            # Validation & logging
            log_dict = {'step': it, 'train_loss': loss.item() * gradient_accum, 'lr': lr, 'elapsed': time.time()-start_time}
            if it % val_interval == 0:
                val_loss = evaluate(model, val_data, batch_size, context_length, device)
                ppl = perplexity(
                    logits.view(-1, V),   
                    yb.reshape(-1)        
                ).item()
                log_dict.update({'val_loss': val_loss, 'ppl': ppl})
            if it % log_interval == 0 or it % val_interval == 0:
                wandb.log(log_dict, step=it)
                print(f"Step {it}: train_loss={log_dict['train_loss']:.4f}, "
                    f"val_loss={log_dict.get('val_loss', float('nan')):.4f}, "
                    f"ppl={log_dict.get('ppl', float('nan')):.2f}, lr={lr:.2e}")
            # Checkpoint
            if run_ckpt_path is not None and it % save_interval == 0:
                save_checkpoint(model, optimizer, it, os.path.join(run_ckpt_path, f"checkpoint_{it}.pt"))

                print(f"Saved checkpoint at iter {it}")

    # Final save
    if run_ckpt_path is not None:
        save_checkpoint(model, optimizer, it, os.path.join(run_ckpt_path, f"checkpoint_{it}.pt"))
    print(f"Training complete at step {it}")
    wandb.finish()


def run_decode(cfg):
    # model and decoding parameters from config
    device = torch.device(cfg.get('device', 'cpu'))
    # instantiate model
    model = TransformerLM(
        d_model=cfg['d_model'],
        num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'],
        weights={},
        vocab_size=cfg['vocab_size'],
        context_length=cfg['context_length'],
        num_layers=cfg['num_layers'],
        theta=None,
        token_positions=None,
        max_seq_len=cfg['context_length']
    ).to(device)
    # load checkpoint
    load_checkpoint(cfg['checkpoint_path'], model, torch.optim.AdamW(model.parameters()))
    # tokenizer
    tok = Tokenizer.from_files(
        cfg['vocab_file'],
        cfg['merges_file'],
        special_tokens=cfg.get('special_tokens', None)
    )
    prompt_ids = tok.encode(cfg.get('prompt', 'Once upon a time,'))
    eot = tok.encode(cfg.get('end_token', '<|endoftext|>'))[0]

    # sampling args
    max_tokens = cfg.get('decode_max_tokens', 100)
    temperature= cfg.get('decode_temperature', 1.0)
    top_p      = cfg.get('decode_top_p', 0.9)


    ids, text = decode(
        model,
        tok,
        input_prompt=prompt_ids,
        end_token_id=eot,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    print(text)

if __name__ == '__main__':
    # usage: python script.py train --config config.yaml
    if len(sys.argv) >= 4 and sys.argv[1] in ('train','decode') and sys.argv[2]=='--config':
        mode = sys.argv[1]
        cfg_path = sys.argv[3]
        with open(f"model_configs/{cfg_path}") as f:
            cfg = yaml.safe_load(f)
        
        # argv:  python train_lm.py train --config cfg.yaml key1=val1 key2=val2 ...
        for override in sys.argv[4:]:
            if "=" not in override:
                raise ValueError(f"Bad override '{override}', expected key=value")
            k, v = override.split("=", 1)
            if k in cfg:
                cfg[k] = type(cfg[k])(v)       # cast to original type
            else:
                cfg[k] = yaml.safe_load(v)     # new key → best-effort cast


        # Automatically patch paths for GPU cluster if needed
        for k in ['train_data', 'val_data', 'checkpoint_path', 'vocab_file', 'merges_file']:
            if k in cfg:
                cfg[k] = get_cluster_data_path(cfg[k])

        if mode == 'train':
            train_lm(cfg)
        else:
            run_decode(cfg)
    else:
        print(f"Usage: {sys.argv[0]} <train|decode> --config CONFIG_YAML_PATH")
