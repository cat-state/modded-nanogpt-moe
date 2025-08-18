import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
import math
import gc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


def hash_select(token_ids, num_experts, null_expert_bias=0.0):
    expert_idx = (token_ids[..., None].float() % num_experts).to(token_ids.dtype)
    routing_weights = torch.nn.functional.one_hot(expert_idx, num_classes=num_experts)
    selected_prob, selected_expert = torch.max(routing_weights, dim=-1, keepdim=True)

    if null_expert_bias > 0:
        selected_prob = selected_prob / (selected_prob + null_expert_bias)
    return selected_expert, routing_weights, selected_prob

def expert_mixing(x, experts, expert_ids, gate):
    B, T, C = x.shape
    x = x.reshape(B * T, C)
    gate = gate.reshape(B * T, -1)
    expert_ids = expert_ids.reshape(B * T, -1)
    output = torch.zeros_like(x)
    
    for i in range(len(experts)):
        mask = (expert_ids == i)
        rows, cols = torch.nonzero(mask, as_tuple=True)
        if len(rows) == 0:
            continue
        tokens = x[rows]
        expert_out = experts[i](tokens)
        weights = gate[rows, cols].unsqueeze(1)
        output.index_add_(0, rows, expert_out * weights)

    return output.reshape(B, T, C)


class LearnedRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, null_expert_bias=0.0):
        super().__init__()
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.top_k = top_k
        self.null_expert_bias = null_expert_bias
    
    def forward(self, x):
        logits = self.router(x)
        probs = logits.softmax(dim=-1)
        gate, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        gate = gate / (gate.sum(dim=-1, keepdim=True) + self.null_expert_bias)
        return topk_idx, probs, gate

def aux_loss(probs, expert_ids):
    num_experts = probs.size(-1)
    counts = torch.bincount(
        expert_ids.flatten(),
        minlength=num_experts
    ).float()
    actual = counts / counts.sum()
    expected = probs.reshape(-1, num_experts).mean(0)
    return num_experts * (actual * expected).sum(), actual


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = 4
        self.top_k       = 1
        assert 1 <= self.top_k <= self.num_experts, "`k` must be in [1, #experts]"
        self.router_type  = 'learned'
        self.null_expert_bias = 0.0
        assert self.router_type in ('hash', 'learned')

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        
        if self.router_type == 'learned':
            self.router  = LearnedRouter(config.n_embd, self.num_experts, self.top_k, self.null_expert_bias)

    def forward(self, x, token_idx=None):

        if self.router_type == "learned":
            topk_idx, probs, gate = self.router(x)
        elif self.router_type == "hash":
            topk_idx, probs, gate = hash_select(token_idx, self.num_experts)
            gate = gate.to(x.dtype)
            probs = probs.to(x.dtype)
        else:
            raise ValueError(f"unknown routing type: {self.router_type}")
        B, T, C = x.shape
        BT      = B * T
        x_flat  = x.reshape(BT, C)
        y_flat  = torch.zeros_like(x_flat)

        probs_flat = probs.reshape(BT, -1)
        gate_flat = gate.reshape(BT, self.top_k)
        idx_flat  = topk_idx.reshape(BT, self.top_k)

        y = expert_mixing(x, self.experts, topk_idx, gate)

        # aux loss and router statistics

        # token-wise entropy of router distribution (normalized to [0,1])
        eps = 1e-9
        with torch.no_grad():
            token_H = -(probs_flat * (probs_flat + eps).log()).sum(-1)
            router_entropy = token_H.mean() / math.log(float(self.num_experts))
        # learned paper:  L_aux = E * <load,prob>
        
        aux, frac = aux_loss(probs, idx_flat)
        if self.router_type == "hash":
            aux = torch.tensor(0.0, device=x.device, requires_grad=self.training)
        else:
            pass

        return y, aux, router_entropy, frac


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MoE(config)

    def forward(self, x, token_idx=None):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        mlp_out, aux, router_entropy, expert_balance = self.mlp(F.rms_norm(x, (x.size(-1),)), token_idx)
        x = x + mlp_out
        return x, aux, router_entropy, expert_balance

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True, aux_coeff=0.0):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        total_aux = 0
        total_router_entropy = 0
        total_expert_balance = None
        per_layer_router_entropy = []
        per_layer_expert_balance = []
        for block in self.transformer.h:
            x, aux, router_entropy, expert_balance = block(x, idx)
            total_aux = total_aux + aux
            total_router_entropy = total_router_entropy + router_entropy
            if total_expert_balance is None:
                total_expert_balance = expert_balance
            else:
                total_expert_balance = total_expert_balance + expert_balance
            per_layer_router_entropy.append(router_entropy)
            per_layer_expert_balance.append(expert_balance)
        x = F.rms_norm(x, (x.size(-1),))

        # average stats across blocks
        num_blocks = len(self.transformer.h)
        avg_router_entropy = total_router_entropy / num_blocks
        avg_expert_balance = total_expert_balance / num_blocks
        layer_router_entropy = torch.stack(per_layer_router_entropy)
        layer_expert_balance = torch.stack(per_layer_expert_balance, dim=0)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if aux_coeff > 0.0:
                loss = loss + total_aux * aux_coeff
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss, total_aux, avg_router_entropy, avg_expert_balance, layer_router_entropy, layer_expert_balance

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 32 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 4578 # number of iterations to run
    warmup_iters : int = 0
    warmdown_iters : int = 1308 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0.0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = model.module # always contains the "raw" unwrapped model
num_experts = raw_model.transformer.h[0].mlp.num_experts
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True)
enable_flash_sdp(False)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

# init the optimizer(s)
all_h_params = list(raw_model.transformer.h.parameters())
muon_params = all_h_params
optimizer1 = torch.optim.AdamW([raw_model.transformer.wte.weight], lr=0.3,   betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
optimizer2 = torch.optim.AdamW([raw_model.lm_head.weight], lr=0.002, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)

optimizer3 = Muon(muon_params,                                     lr=0.02,  momentum=0.95)
optimizers = [optimizer1, optimizer2, optimizer3]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')
    # init wandb
    wandb_project = "modded-nanogpt-moe"
    wandb_run = wandb.init(project=wandb_project, name=run_id, config={
        'data': {
            'input_bin': args.input_bin,
            'input_val_bin': args.input_val_bin,
            'val_tokens': args.val_tokens,
        },
        'training': {
            'batch_size_total': args.batch_size,
            'device_batch_size': args.device_batch_size,
            'sequence_length': args.sequence_length,
            'num_iterations': args.num_iterations,
            'accumulation_steps': train_accumulation_steps,
            'val_steps': val_steps,
            'save_every': args.save_every,
        },
        'scheduler': {
            'type': 'linear_warmup_constant_linear_warmdown',
            'warmup_iters': args.warmup_iters,
            'warmdown_iters': args.warmdown_iters,
        },
        'optimizers': {
            'embed': {
                'type': 'AdamW',
                'lr': optimizer1.param_groups[0]['lr'],
                'betas': tuple(optimizer1.param_groups[0]['betas']),
                'fused': bool(optimizer1.param_groups[0].get('fused', False)),
                'weight_decay': args.weight_decay,
            },
            'head': {
                'type': 'AdamW',
                'lr': optimizer2.param_groups[0]['lr'],
                'betas': tuple(optimizer2.param_groups[0]['betas']),
                'fused': bool(optimizer2.param_groups[0].get('fused', False)),
                'weight_decay': args.weight_decay,
            },
            'muon_blocks': {
                'type': 'Muon',
                'lr': optimizer3.param_groups[0]['lr'],
                'momentum': optimizer3.defaults.get('momentum', None),
                'nesterov': optimizer3.defaults.get('nesterov', None),
                'backend': optimizer3.defaults.get('backend', None),
                'backend_steps': optimizer3.defaults.get('backend_steps', None),
            }
        },
        'model': {
            'vocab_size': num_vocab,
            'n_layer': raw_model.config.n_layer,
            'n_head': raw_model.config.n_head,
            'n_embd': raw_model.config.n_embd,
            'num_experts': num_experts,
            'router_type': raw_model.transformer.h[0].mlp.router_type,
            'router_top_k': raw_model.transformer.h[0].mlp.top_k,
            'null_expert_bias': raw_model.transformer.h[0].mlp.null_expert_bias,
        },
        'loss': {
            'type': 'cross_entropy',
            'ignore_index': -1,
            'aux_coeff_train': 0.01,
            'aux_coeff_val': 0.0,
        },
        'dist': {
            'world_size': ddp_world_size,
            'rank': ddp_rank,
            'local_rank': ddp_local_rank,
        },
        'precision': {
            'amp_dtype': 'bfloat16',
        },
        'torch': {
            'compile': True,
            'attention_backend': 'cudnn_sdp',
        },
    })

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        val_router_entropy = torch.tensor(0.0, device=device)
        val_expert_balance = torch.zeros(num_experts, device=device)
        # per-layer
        n_layers = raw_model.config.n_layer
        val_layer_router_entropy = torch.zeros(n_layers, device=device)
        val_layer_expert_balance = torch.zeros(n_layers, num_experts, device=device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                with ctx:
                    _, loss, total_aux, router_entropy, expert_balance, layer_router_entropy, layer_expert_balance = model(x_val, y_val, return_logits=False, aux_coeff=0.0)
                    val_loss += loss.detach()
                    val_router_entropy = val_router_entropy + router_entropy.detach()
                    val_expert_balance = val_expert_balance + expert_balance.detach()
                    val_layer_router_entropy = val_layer_router_entropy + layer_router_entropy.detach()
                    val_layer_expert_balance = val_layer_expert_balance + layer_expert_balance.detach()
                    del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # average and all-reduce router stats
        val_router_entropy = val_router_entropy / val_steps
        val_expert_balance = val_expert_balance / val_steps
        val_layer_router_entropy = val_layer_router_entropy / val_steps
        val_layer_expert_balance = val_layer_expert_balance / val_steps
        dist.all_reduce(val_router_entropy, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_expert_balance, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_layer_router_entropy, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_layer_expert_balance, op=dist.ReduceOp.AVG)
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # compute router grad norms (CE and AUX separately) occasionally at validation interval
        # Use a single micro-batch to avoid heavy cost
        model.train()
        # grab a fresh batch (train or val doesn't matter for grads inspection)
        x_probe, y_probe = val_loader.next_batch()
        # 1) CE-only
        model.zero_grad(set_to_none=True)
        gc.collect()
        with ctx:
            _, loss_ce, total_aux_probe, _, _, _, _ = model(x_probe, y_probe, return_logits=False, aux_coeff=0.0)
        loss_ce.backward()
        ce_router_layer_grad_norms = []
        for li in range(raw_model.config.n_layer):
            if raw_model.transformer.h[li].mlp.router_type == "learned":
                p = raw_model.transformer.h[li].mlp.router.router.weight
                gnorm = p.grad.detach().float().norm(2) if p.grad is not None else torch.tensor(0.0, device=device)
                ce_router_layer_grad_norms.append(gnorm)
            else:
                ce_router_layer_grad_norms.append(torch.tensor(0.0, device=device))
        ce_router_layer_grad_norms = torch.stack(ce_router_layer_grad_norms)
        dist.all_reduce(ce_router_layer_grad_norms, op=dist.ReduceOp.AVG)
        # 2) AUX-only
        model.zero_grad(set_to_none=True)
        gc.collect()
        with ctx:
            _, _, total_aux_probe, _, _, _, _ = model(x_probe, y_probe, return_logits=False, aux_coeff=0.0)
        # Backprop aux explicitly
        total_aux_probe.backward()
        aux_router_layer_grad_norms = []
        for li in range(raw_model.config.n_layer):
            if raw_model.transformer.h[li].mlp.router_type == "learned":
                p = raw_model.transformer.h[li].mlp.router.router.weight
                gnorm = p.grad.detach().float().norm(2) if p.grad is not None else torch.tensor(0.0, device=device)
                aux_router_layer_grad_norms.append(gnorm)
            else:
                aux_router_layer_grad_norms.append(torch.tensor(0.0, device=device))
        aux_router_layer_grad_norms = torch.stack(aux_router_layer_grad_norms)
        dist.all_reduce(aux_router_layer_grad_norms, op=dist.ReduceOp.AVG)
        # zero out any probe grads
        model.zero_grad(set_to_none=True)
        gc.collect()
        # log to wandb
        if master_process:
            wandb_log_extra = {}
            for li in range(raw_model.config.n_layer):
                wandb_log_extra[f'Router Grad Norms (CE)/Layer {li}'] = float(ce_router_layer_grad_norms[li].item())
                wandb_log_extra[f'Router Grad Norms (AUX)/Layer {li}'] = float(aux_router_layer_grad_norms[li].item())
            wandb.log(wandb_log_extra, step=step)
        # now also log the earlier val metrics
        if master_process:
            wandb_log = {
                'val/loss': float(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss),
                'val/router_entropy': float(val_router_entropy.item()),
            }
            for i in range(num_experts):
                wandb_log[f'val/expert_balance/{i}'] = float(val_expert_balance[i].item())
            for li in range(n_layers):
                wandb_log[f'Router Entropy/Layer {li}'] = float(val_layer_router_entropy[li].item())
                for ei in range(num_experts):
                    wandb_log[f'Expert Balance/Layer {li}/{ei}'] = float(val_layer_expert_balance[li, ei].item())
            wandb_log['train/time_ms'] = float(training_time_ms)
            wandb_log['train/step_avg_ms'] = float(training_time_ms/(timed_steps-1))
            wandb.log(wandb_log, step=step)

        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    router_entropy_sum = torch.tensor(0.0, device=device)
    expert_balance_sum = torch.zeros(num_experts, device=device)
    # per-layer accumulators
    n_layers = raw_model.config.n_layer
    layer_router_entropy_sum = torch.zeros(n_layers, device=device)
    layer_expert_balance_sum = torch.zeros(n_layers, num_experts, device=device)
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss, total_aux, router_entropy, expert_balance, layer_router_entropy, layer_expert_balance = model(x, y, return_logits=False, aux_coeff=0.01)
            train_loss = loss.detach()
            router_entropy_sum = router_entropy_sum + router_entropy.detach()
            expert_balance_sum = expert_balance_sum + expert_balance.detach()
            layer_router_entropy_sum = layer_router_entropy_sum + layer_router_entropy.detach()
            layer_expert_balance_sum = layer_expert_balance_sum + layer_expert_balance.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n)
    for p in model.parameters():
        p.grad /= train_accumulation_steps

    # compute gradient norm (after accumulation average, before optimizer step)
    grad_norm = torch.tensor(0.0, device=device)
    grads_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grads_norms.append(p.grad.detach().float().norm(2))
    if len(grads_norms) > 0:
        grad_norm = torch.norm(torch.stack(grads_norms), 2)

    # average and all-reduce router stats across accumulation steps and processes
    router_entropy_avg = router_entropy_sum / train_accumulation_steps
    expert_balance_avg = expert_balance_sum / train_accumulation_steps
    layer_router_entropy_avg = layer_router_entropy_sum / train_accumulation_steps
    layer_expert_balance_avg = layer_expert_balance_sum / train_accumulation_steps
    dist.all_reduce(router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(expert_balance_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_expert_balance_avg, op=dist.ReduceOp.AVG)

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        # wandb logging
        wandb_log = {
            'train/loss': float(train_loss.item()),
            'train/router_entropy': float(router_entropy_avg.item()),
            'train/grad_norm': float(grad_norm.item()),
            'train/step_time_ms': float(approx_time),
            'train/step_avg_ms': float(approx_time/timed_steps),
        }
        for i_exp in range(num_experts):
            wandb_log[f'train/expert_balance/{i_exp}'] = float(expert_balance_avg[i_exp].item())
        # per-layer router stats (no per-step grad norms anymore)
        for li in range(n_layers):
            wandb_log[f'Router Entropy/Layer {li}'] = float(layer_router_entropy_avg[li].item())
            for ei in range(num_experts):
                wandb_log[f'Expert Balance/Layer {li}/{ei}'] = float(layer_expert_balance_avg[li, ei].item())
        wandb.log(wandb_log, step=step+1)

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    try:
        wandb.finish()
    except Exception:
        pass

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
