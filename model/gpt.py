from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import PreTrainedModel


@dataclass
class ModelArgs:
    context_size = 1024
    vocab_size = 50304
    n_embd = 768
    n_layer = 12
    n_head = 12
    dropout = 0.0
    layer_norm_epsilon = 1e-5
    bias = False
    initializer_range = 0.01


class Head(nn.Module):
    def __init__(self, args: ModelArgs, h_size: int):
        super().__init__()

        self.key = nn.Linear(args.n_embd, h_size, bias=args.bias)
        self.query = nn.Linear(args.n_embd, h_size, bias=args.bias)
        self.values = nn.Linear(args.n_embd, h_size, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)

        # Attention Mask
        self.register_buffer("bias", torch.tril(torch.ones(args.context_size, args.context_size)))

    def forward(self, x: torch.Tensor):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        k = self.key(x)
        q = self.query(x)
        v = self.values(x)

        # Attention
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        att = att.masked_fill(self.bias[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, T, hs)

        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        head_size = args.n_embd // args.n_head
        self.heads = nn.ModuleList([Head(args, head_size) for _ in range(args.n_head)])
        self.projection = nn.Linear(head_size * args.n_head, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor):
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.dropout(self.projection(y))
        return y


class MLP(nn.Module):  # Feed forward
    def __init__(self, args: ModelArgs):
        super().__init__()

        n_inner = 4 * args.n_embd  # Dimensionality of the inner feed-forward layer
        self.seq = nn.Sequential(
            nn.Linear(args.n_embd, n_inner, bias=args.bias),
            nn.GELU(),
            nn.Linear(n_inner, args.n_embd, bias=args.bias),  # projection layer
            nn.Dropout(args.dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln_1 = nn.LayerNorm(args.n_embd)
        self.mha = MultiHeadAttention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd)
        self.mlp = MLP(args)

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module, PreTrainedModel):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.token_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.positional_encoding = nn.Embedding(args.context_size, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([Block(args) for _ in range(args.n_layer)])
        self.norm = nn.LayerNorm(args.n_embd, args.bias)

        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=args.bias)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = self.args.initializer_range

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, tokens: torch.LongTensor, targets=None):
        if tokens is None:
            raise ValueError("You must provide the tokens to forward the training data")

        device = tokens.device
        B, T = tokens.size()
        assert (
            T <= self.args.context_size
        ), f"The sequence size ({T}) must be less than the context size ({self.context_size})"

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        token_emb = self.token_embeddings(tokens)
        positional_enc = self.positional_encoding(pos)
        h = self.dropout(token_emb + positional_enc)

        for layers in self.layers:
            h = layers(h)
        h = self.norm(h)
        logits = self.output(h)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None

        return loss, logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, top_k=None, temperature=1.0):
        """Generate text based on an initial input sequence."""
        for _ in range(max_new_tokens):
            input_ids_cond = (
                input_ids
                if input_ids.size(1) <= self.args.context_size
                else input_ids[:, -self.args.context_size :]
            )

            _, logits = self(input_ids_cond)

            logits = logits[:, -1, :]  # (B, C)

            if top_k is not None:
                # Top-k sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_k / 100.0
                # Shift the indices to account for the last token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float("Inf")

            # Apply temperature to the logits
            logits /= temperature

            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            input_ids_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

        return input_ids
