# Code heavily borrowed from https://github.com/keiserlab/exceiver/tree/main/exceiver
# From paper https://arxiv.org/abs/2210.14330
# other possible data from: https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219
# other public covid data: https://onlinelibrary.wiley.com/doi/10.1002/ctd2.104

from typing import Dict, Optional, Tuple
import torch
from torch import nn


class CrossAttn(nn.Module):
    def __init__(self, query_dim: int, key_val_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layernorm_kv = nn.LayerNorm(key_val_dim)
        self.layernorm_q = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=dropout,
            kdim=key_val_dim,
            vdim=key_val_dim,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            *[
                nn.LayerNorm(query_dim),
                nn.Linear(query_dim, query_dim),
                nn.GELU(),
                nn.LayerNorm(query_dim),
                nn.Linear(query_dim, query_dim),
            ]
        )

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


class ProcessSelfAttn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            n_heads,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.transformer(latent)


class Exceiver(nn.Module):

    # seq_dim: Dimension of gene representations.
    # query_len: Size of input query, or latent representation length.
    # query_dim: Dimension of input query.
    # n_layers: Number of ProcessSelfAttention layers.
    # n_heads: Number of ProcessSelfAttention heads.
    # dim_feedforward: Dimension of ProcessSelfAttention feedforward network.
    # dropout: Value of ProcessSelfAttention dropout.

    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        query_len: int,
        query_dim: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        class_dist: Optional[Dict[str, float]] = None,
        dropout: float = 0.0,  # for the process attention module
        rank_order: bool = False,
        **kwargs,
    ):

        # Initialize superclass
        super().__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        # rank ordering will change how genes are embedded
        self.rank_order = rank_order
        # create the gene embeddings based on whether to use rank ordering
        self._create_gene_embeddings()

        # Embeddings and attention blocks
        self.n_classes = len(class_dist) if class_dist is not None else 0

        # self.output_emb = nn.Embedding(seq_len + 1, seq_dim, padding_idx=seq_len)
        self.query_emb = nn.Parameter(torch.randn(query_len, query_dim))
        self.encoder_cross_attn = CrossAttn(
            query_dim, seq_dim, num_heads=n_heads,
        )
        self.process_self_attn = ProcessSelfAttn(
            query_dim, n_layers, n_heads, dim_feedforward, dropout,
        )
        self.decoder_cross_attn = CrossAttn(
            seq_dim, query_dim
        )  # query is now gene embedding

        self.gene_mlp = nn.Sequential(
            nn.LayerNorm(seq_dim), nn.Linear(seq_dim, 1)
        )

        self.class_dist = class_dist
        if self.n_classes > 0:
            self.class_emb = nn.Embedding(self.n_classes + 1, seq_dim, padding_idx=self.n_classes)
            self.class_mlp = nn.ModuleDict(
                {k: nn.Sequential(nn.LayerNorm(seq_dim), nn.Linear(seq_dim, len(v))
                    ) for k, v in class_dist.items()
                }
            )

    def encoder_attn_step(
        self,
        key_vals: torch.Tensor,
        input_query: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        latent, encoder_weights = self.encoder_cross_attn(
            input_query, key_vals, key_padding_mask
        )
        return latent, encoder_weights

    def _create_gene_embeddings(self):

        if self.rank_order:
            self.gene_emb_low = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_emb_high = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
        else:
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_w = nn.Parameter(torch.ones(1, self.seq_len - 1))
            self.gene_val_b = nn.Parameter(torch.zeros(1, self.seq_len - 1))

    def _gene_embedding(self, gene_ids: torch.Tensor, gene_vals: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.rank_order:
            if gene_vals is not None:
                return (1 - gene_vals) * self.gene_emb_low(gene_ids) + gene_vals * self.gene_emb_high(gene_ids)
            else:
                return self.gene_emb_high(gene_ids)
        else:
            if gene_vals is not None:
                alpha = gene_vals * self.gene_val_w + self.gene_val_b
                gene_emb = self.gene_emb(gene_ids)
                return alpha.unsqueeze(2) * gene_emb
            else:
                return self.gene_emb(gene_ids)


    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_target_ids: torch.Tensor,
        class_target_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)

        class_target_query = self.class_emb(class_target_ids)

        key_vals = self._gene_embedding(gene_ids, gene_vals)
        gene_target_query = self._gene_embedding(gene_target_ids)

        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )
        latent = self.process_self_attn(latent)

        # Query genes and classes
        decoder_out, decoder_weights = self.decoder_cross_attn(
            torch.cat((gene_target_query, class_target_query), dim=1),
            latent,
            key_padding_mask=None,
        )

        # Predict genes
        n_genes = gene_target_query.size(1)
        gene_pred = self.gene_mlp(decoder_out[:, : n_genes, :])

        # Predict classes
        if self.n_classes > 0:
            class_pred = {}
            for n, k in enumerate(self.class_mlp.keys()):
                class_pred[k] = self.class_mlp[k](decoder_out[:, n_genes + n, :])
        else:
            class_pred = None

        return gene_pred, class_pred, latent


def extract_state_dict(model_save_path, device):

    state_dict = {}
    ckpt = torch.load(model_save_path)
    for k, v in ckpt["state_dict"].items():
        k1 = k.split(".")
        k1 = ".".join(k1[1:])
        state_dict[k1] = v.to(device=device)

    return state_dict
