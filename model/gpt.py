import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
context_size = 12
vocab_size = 50257
#n_positions = 1024
n_embd = 128
n_layer = 4
n_head = 4
dropout = 0.1
#layer_norm_epsilon = 1e-5
bias = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module): 

    def __init__(self, n_embd, h_size, bias):
        super().__init__()
        
        self.key = nn.Linear(n_embd, h_size, bias=bias)
        self.query = nn.Linear(n_embd, h_size, bias=bias)
        self.values = nn.Linear(n_embd, h_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        # Attention Mask
        self.register_buffer('bias', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        k = self.key(x)
        q = self.query(x)
        v = self.values(x)

        # Attention
        att = (q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5)
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v # (B, T, hs)

        return y
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias):
        super().__init__()

        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, bias) for _ in range(n_head)])
        self.projection = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = torch.cat([head(x) for head in self.heads], dim=-1)
        y = self.dropout(self.projection(y))
        return y


class MLP(nn.Module): # Feed forward
    
    def __init__(self, n_embd, dropout, bias):
        super().__init__()

        n_inner = 4 * n_embd # Dimensionality of the inner feed-forward layer
        self.seq = nn.Sequential(
            nn.Linear(n_embd, n_inner, bias=bias),
            nn.GELU(),
            nn.Linear(n_inner, n_embd, bias=bias), # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.seq(x)
    

class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(vocab_size, n_embd),
            positional_encoding = nn.Embedding(context_size, n_embd),
            dropout = nn.Dropout(dropout),
            blocks = nn.ModuleList([Block(n_embd, n_head, bias) for _ in range(n_layer)]),
            final_ln = nn.LayerNorm(n_embd, bias),
        ))
        self.l_output_head = nn.Linear(n_embd, vocab_size, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        B, T = input_ids.size()
        assert  T <= context_size, f'The sequence size ({T}) must be less than the context size ({context_size})'

        pos = torch.arange(0, T, dtype=torch.long, device=device) 

        token_emb = self.transformer.token_embedding(input_ids)
        positional_enc = self.transformer.positional_encoding(pos)
        x = self.transformer.dropout(token_emb + positional_enc)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.final_ln(x)
        logits = self.l_output_head(x)
        
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None

        return logits, loss

    def generate(self, input_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            input_ids_cond = input_ids if input_ids.size(1) <= context_size else input_ids[:, -context_size:]

            logits, loss = self(input_ids_cond)

            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            input_ids_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

        return input_ids
