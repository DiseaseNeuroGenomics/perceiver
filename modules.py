
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.fft

class MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MLPEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        n_input: int = 1,
        linear: bool = False,
        n_hidden: int = 32,
    ):
        super().__init__()
        print("Creating gene value embedding")

        self.n_input = n_input
        if linear:
            self.mlp = nn.Linear(n_input, embedding_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_input, n_hidden),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(n_hidden, embedding_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = x.unsqueeze(-1) if self.n_input == 1 else x
        return self.mlp(x)


class SimpleHyenaBlock(nn.Module):
    def __init__(self, dim, kernel_size=64, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim or 4 * dim

        self.modulator = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

        # One kernel per channel
        self.kernel = nn.Parameter(torch.randn(dim, kernel_size))

        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        B, L, D = x.shape
        x_proj = self.in_proj(x)
        mod = self.modulator(x)

        modulated = x_proj * mod  # elementwise modulation

        # FFT-based convolution per channel
        x_fft = torch.fft.rfft(modulated.transpose(1, 2), n=L)
        k_fft = torch.fft.rfft(self.kernel, n=L)
        y_fft = x_fft * k_fft.unsqueeze(0)  # broadcasting kernel
        y = torch.fft.irfft(y_fft, n=L).transpose(1, 2)

        return self.out_proj(y)


class EncoderDeocderStack(nn.Module):

    def __init__(self, query_dim: int, gene_emb_dim: int, n_layers: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderDeocder(query_dim, gene_emb_dim, num_heads=num_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        latent_query: torch.Tensor,
        gene_query: torch.Tensor,
        key_val: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for n, layer in enumerate(self.layers):
            mask = key_padding_mask if n == 0 else None
            key_val, _ = layer(latent_query, gene_query, key_val, mask)

        return key_val


class EncoderDeocder(nn.Module):

    def __init__(self, query_dim: int, gene_emb_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layernorm_kv = nn.LayerNorm(gene_emb_dim)
        self.layernorm_q = nn.LayerNorm(query_dim)

        self.layernorm_g = nn.LayerNorm(gene_emb_dim)
        self.layernorm_l = nn.LayerNorm(query_dim)

        self.encoder = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=0.0,
            kdim=gene_emb_dim,
            vdim=gene_emb_dim,
            batch_first=True,
        )

        self.decoder= nn.MultiheadAttention(
            gene_emb_dim,
            num_heads,
            dropout=0.0,
            kdim=gene_emb_dim,
            vdim=gene_emb_dim,
            batch_first=True,
        )

        self.mlp0 = MLP(query_dim, query_dim, dropout=dropout)
        self.mlp1 = MLP(gene_emb_dim, gene_emb_dim, dropout=dropout)

    def forward(
        self,
        latent_query: torch.Tensor,
        gene_query: torch.Tensor,
        key_val: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        residual: bool = True,
    ) -> torch.Tensor:

        norm_kv = self.layernorm_kv(key_val)
        norm_q = self.layernorm_q(latent_query)

        # Encoder: full input -> latent
        latent, _ = self.encoder(norm_q, norm_kv, norm_kv, key_padding_mask)
        # latent = latent_query + latent_attn
        latent = latent + self.mlp0(latent)

        # Decoder: latent -> full
        norm_g = self.layernorm_g(gene_query)
        norm_l = self.layernorm_l(latent)

        gene_out, _ = self.decoder(norm_g, norm_l, norm_l)
        # gene_out = gene_query + decoder_out
        gene_out = key_val + self.mlp1(gene_out)

        return gene_out, latent



class CrossAttn(nn.Module):
    def __init__(self, query_dim: int, key_val_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layernorm_kv = nn.LayerNorm(key_val_dim)
        self.layernorm_q = nn.LayerNorm(query_dim)
        print("CrossAttn drop", dropout)
        self.cross_attn = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=0.0,
            kdim=key_val_dim,
            vdim=key_val_dim,
            batch_first=True,
        )
        self.mlp = MLP(query_dim, query_dim, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_val: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        residual: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_k = self.layernorm_kv(key_val)
        norm_v = self.layernorm_kv(key_val)
        norm_q = self.layernorm_q(query)

        # TODO: do I need to output weights??
        latent, weights = self.cross_attn(norm_q, norm_k, norm_v, key_padding_mask)
        # residual connection
        if residual:
            latent = latent + query
        latent = self.mlp(latent) + latent
        return latent, weights


class DecoderNoCrossAttn(nn.Module):

    def __init__(
        self,
        seq_dim: int,
        dropout: float = 0.0,  # for the process attention module
        n_out: int = 1,
        layernorm: bool = True,
        hidden_expansion: float = 1,
        hidden_layers: int = 1,
    ):

        super().__init__()

        hidden_dim = int(hidden_expansion * seq_dim)

        seq = [
            nn.LayerNorm(seq_dim) if layernorm else nn.Identity(),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for n in range(1, hidden_layers):
            seq += [
                nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        output_layer = nn.Linear(hidden_dim, n_out)
        if n_out == 3:
            # Now manually initialize:
            # nn.init.xavier_uniform_(output_layer.weight, gain=0.1)  # or another sensible init
            nn.init.zeros_(output_layer.bias)

            # Then set pi bias to a negative value
            with torch.no_grad():
                output_layer.bias[2].fill_(-3.0)  # Or maybe -3.0 for even smaller dropout

        seq += [output_layer]

        self.gene_mlp = nn.Sequential(*seq)



    def forward(self, x: torch.Tensor):

        gene_pred = self.gene_mlp(x)

        return gene_pred


class Decoder(nn.Module):

    def __init__(
        self,
        seq_dim: int,
        query_dim: int,
        cross_attn_dropout: float = 0.0,
        dropout: float = 0.0,  # for the process attention module
        n_out: int = 1,
        layernorm: bool = True,
        hidden_expansion: float = 1,
        hidden_layers: int = 1,
    ):

        super().__init__()

        self.decoder_cross_attn = CrossAttn(
            seq_dim,
            query_dim,
            dropout=cross_attn_dropout,
        )  # query is now gene embedding

        hidden_dim = int(hidden_expansion * seq_dim)

        seq = [
            nn.LayerNorm(seq_dim) if layernorm else nn.Identity(),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for n in range(1, hidden_layers):
            seq += [
                nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        output_layer = nn.Linear(hidden_dim, n_out)
        if n_out == 3:
            # Now manually initialize:
            # nn.init.xavier_uniform_(output_layer.weight, gain=0.1)  # or another sensible init
            nn.init.zeros_(output_layer.bias)

            # Then set pi bias to a negative value
            with torch.no_grad():
                output_layer.bias[2].fill_(-3.0)  # Or maybe -3.0 for even smaller dropout

        seq += [output_layer]

        self.gene_mlp = nn.Sequential(*seq)



    def forward(self, latent: torch.Tensor, gene_query: torch.Tensor):

        # Query genes and cell properties
        # Decoder out will contain the latent for both genes and cell properties, concatenated together
        decoder_out, _ = self.decoder_cross_attn(
            gene_query,
            latent,
            key_padding_mask=None,
        )

        # Predict genes
        n_genes = gene_query.size(1)
        gene_pred = self.gene_mlp(decoder_out[:, : n_genes, :])

        return gene_pred

class gMLP_stack(nn.Module):
    def __init__(self, seq_len: int, seq_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        self.gmlps = nn.Sequential(
            *[gMLP(seq_len, seq_dim, hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmlps(x)


class hyena_stack(nn.Module):
    def __init__(self, seq_dim: int, n_layers: int, kernel_size: int = 64, hidden_dim: int = 256):
        super().__init__()

        self.hyena_seq = nn.Sequential(
            *[SimpleHyenaBlock(seq_dim, kernel_size=kernel_size, hidden_dim=hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hyena_seq(x)


class gMLP(nn.Module):
    def __init__(self, query_len: int, seq_dim: int, hidden_dim: int):
        super().__init__()

        # inputs is batch X seq_len X seq_dim
        self.ln0 = nn.LayerNorm(seq_dim)
        self.ln1 = nn.LayerNorm(hidden_dim // 2)
        self.linear0 = nn.Linear(seq_dim, hidden_dim)
        self.linear1 = nn.Linear(query_len, query_len)
        self.linear2 = nn.Linear(hidden_dim // 2, seq_dim)
        self.act = nn.GELU()

        torch.nn.init.ones_(self.linear1.bias)
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        x = self.ln0(input)
        x = self.linear0(x)
        x = self.act(x)
        (u, v) = torch.chunk(x, 2, dim=-1)
        v = self.ln1(v)
        v = torch.transpose(v, 2, 1)
        v = self.linear1(v)
        v = torch.transpose(v, 2, 1)
        x = self.linear2(u * v)
        return x + input


class ProcessSelfAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        print("ProcessSelfAttn drop", dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            n_heads,
            dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # NYM June 24
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.transformer(latent)