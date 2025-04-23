# Code heavily borrowed from https://github.com/keiserlab/exceiver/tree/main/exceiver
# From paper https://arxiv.org/abs/2210.14330
# other possible data from: https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219
# other public covid data: https://onlinelibrary.wiley.com/doi/10.1002/ctd2.104

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn
from modules import MLP, ProcessSelfAttn, MLPEmbedding, Decoder, CrossAttn, gMLP_stack


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
        embedding_strategy: Literal["binned", "continuous", "film"] = "continuous",
        dropout: float = 0.0,  # for the process attention module
        n_bins: Optional[int] = None,
        n_out_feature: Optional[int] = None,
        **kwargs,
    ):

        super().__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.n_bins = n_bins
        self.embedding_strategy = embedding_strategy
        assert (
                embedding_strategy != "binned" or (embedding_strategy == "binned" and n_bins is not None),
                "n_bins must be specified if embedding strategy is 'binned'"
        )

        # create the gene embeddings based on whether to use rank ordering
        self._create_gene_embeddings()

        # self.output_emb = nn.Embedding(seq_len + 1, seq_dim, padding_idx=seq_len)
        self.query_emb = nn.Parameter(torch.randn(query_len, query_dim))

        self.encoder_cross_attn = CrossAttn(
            query_dim, seq_dim, num_heads=n_heads,
        )

        self.process_self_attn = ProcessSelfAttn(
            query_dim, n_layers, n_heads, dim_feedforward, dropout,
        )

        self.decoder = Decoder(seq_dim, query_dim, dropout)
        if n_out_feature is not None:
            self.feature_emb = self.query_emb = nn.Parameter(torch.randn(1, query_dim))
            self.feature_decoder = Decoder(seq_dim, query_dim, dropout, n_out=n_out_feature)

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

        if self.embedding_strategy == "binned":
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_bin_emb = nn.Embedding(self.n_bins + 1, self.seq_dim, padding_idx=0)
        elif self.embedding_strategy == "continuous":
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_emb = MLPEmbedding(self.seq_dim)
        elif self.embedding_strategy == "film":
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_scale = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_emb = MLPEmbedding(self.seq_dim)

        self.target_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)

    def _target_embedding(self, gene_ids: torch.Tensor) -> torch.Tensor:

        #return self.gene_emb(gene_ids)
        return self.target_emb(gene_ids)

    def _gene_embedding(
        self,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
    ) -> torch.Tensor:

        if self.embedding_strategy == "binned":
            return self.gene_emb(gene_ids) + self.gene_bin_emb(gene_vals.to(torch.long))
        elif self.embedding_strategy == "continuous":
            return self.gene_emb(gene_ids) + self.gene_val_emb(gene_vals)
        elif self.embedding_strategy == "film":
            return self.gene_emb(gene_ids) + self.gene_scale(gene_ids) * self.gene_val_emb(gene_vals)


    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_target_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        decode_feature: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)

        key_vals = self._gene_embedding(gene_ids, gene_vals)
        gene_query = self._target_embedding(gene_target_ids)

        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )
        #latent = key_vals
        #print(latent.shape)
        latent = self.process_self_attn(latent)

        gene_pred = self.decoder(latent, gene_query)

        if decode_feature:
            feature_query = self.feature_emb.repeat(len(gene_ids), 1, 1)
            feature_pred = self.feature_decoder(latent, feature_query)
        else:
            feature_pred = None

        return gene_pred, latent, feature_pred


class ATACDecoder(nn.Module):

    def __init__(
        self,
        seq_dim: int,
        query_dim: int,
        atac_window: int,
        dropout: float = 0.0,  # for the process attention module
    ):

        super().__init__()

        self.decoder_cross_attn = CrossAttn(
            seq_dim, query_dim
        )  # query is now gene embedding

        self.atac_mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_dim, atac_window),
        )

    def forward(self, latent: torch.Tensor, atac_query: torch.Tensor):

        # Query genes and cell properties
        # Decoder out will contain the latent for both genes and cell properties, concatenated together
        decoder_out, _ = self.decoder_cross_attn(
            atac_query,
            latent,
            key_padding_mask=None,
        )

        # Predict genes
        n_atac = atac_query.size(1)
        atac_pred = self.atac_mlp(decoder_out[:, : n_atac, :])

        return atac_pred



class ContrastiveGatedMLP(nn.Module):

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
        n_heads: int,
        n_layers: int,
        cell_properties: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,  # for the process attention module
        **kwargs,
    ):

        super().__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        # create the gene embeddings based on whether to use rank ordering
        self._create_gene_embeddings()

        # Embeddings and attention blocks
        self.n_cell_props = len(cell_properties) if cell_properties is not None else 0

        # self.output_emb = nn.Embedding(seq_len + 1, seq_dim, padding_idx=seq_len)
        self.query_emb = nn.Parameter(torch.randn(query_len, query_dim))
        self.cell_emb = nn.Embedding(self.n_cell_props + 1, seq_dim, padding_idx=self.n_cell_props)

        self.encoder_cross_attn = CrossAttn(
            query_dim, seq_dim, num_heads=n_heads,
        )

        self.gmlp = gMLP_stack(query_len, seq_dim, seq_dim * 2, n_layers)

        self._reduce = Reduce(seq_dim, query_len)

        self.mlp = nn.Sequential(
            nn.Linear(seq_dim, seq_dim),
            nn.ReLU(),
            nn.Linear(seq_dim, 128),
        )

        self._cell_predict = SimpleDecoder(seq_dim, cell_properties)

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

        self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
        self.gene_val_w = nn.Embedding(self.seq_len + 1, 1, padding_idx=self.seq_len)
        self.gene_val_b = nn.Embedding(self.seq_len + 1, 1, padding_idx=self.seq_len)

        torch.nn.init.ones_(self.gene_val_w.weight)
        torch.nn.init.zeros_(self.gene_val_b.weight)

    def _gene_embedding(
        self,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
    ) -> torch.Tensor:

        gene_emb = self.gene_emb(gene_ids)
        vals = gene_vals.unsqueeze(2) * self.gene_val_w(gene_ids) +  self.gene_val_b(gene_ids)
        return vals * gene_emb

    def forward(
        self,
        gene_ids: List[torch.Tensor],
        gene_vals: List[torch.Tensor],
        key_padding_mask: List[torch.Tensor],
        n_views: int,
    ):

        z = []
        cell_pred = []

        for n in range(n_views):

            input_query = self.query_emb.repeat(len(gene_ids[n]), 1, 1)
            key_vals = self._gene_embedding(gene_ids[n], gene_vals[n])
            # Main calculation of latent variables
            latent, _ = self.encoder_attn_step(
                key_vals, input_query, key_padding_mask[n]
            )
            latent = self.gmlp(latent)
            h = self._reduce(latent)
            cell_pred.append(self._cell_predict(h.detach()))
            z.append(self.mlp(h))


        return z, cell_pred



class GatedMLP(nn.Module):

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
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,  # for the process attention module
        embedding_strategy: Literal["binned", "continuous", "continuous"] = "continuous",
        linear_embedding: bool = False,
        n_bins: Optional[int] = None,
        cell_properties: Optional[Dict[str, Any]] = None,
        RDA: bool = False,
        **kwargs,
    ):

        super().__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.n_bins = n_bins
        self.embedding_strategy = embedding_strategy
        self.linear_embedding = linear_embedding
        self.RDA = RDA

        assert (
            embedding_strategy != "binned" or (embedding_strategy == "binned" and n_bins is not None),
            "n_bins must be specified if embedding strategy is 'binned'"
        )
        # create the gene embeddings based on whether to use rank ordering
        self._create_gene_embeddings()

        # self.output_emb = nn.Embedding(seq_len + 1, seq_dim, padding_idx=seq_len)
        self.query_emb = nn.Parameter(torch.randn(query_len, query_dim))

        self.encoder_cross_attn = CrossAttn(
            query_dim, seq_dim, num_heads=n_heads,
        )

        self.gmlp = gMLP_stack(query_len, seq_dim, seq_dim * 2, n_layers)
        self.decoder = Decoder(seq_dim, query_dim, dropout)

        self.cell_properties = cell_properties
        if cell_properties is not None:
            n_features = len(cell_properties)
            print("n_features", n_features)

            self.feature_decoder = nn.ModuleDict()
            self.feature_emb = {}
            for k, v in cell_properties.items():
                n_out_feature = len(v["values"])
                print("Feature dim", k, n_out_feature)
                self.feature_emb[k] = nn.Parameter(torch.randn(1, query_dim))
                self.feature_decoder[k] = Decoder(
                    seq_dim,
                    query_dim,
                    dropout=0.5,
                    cross_attn_dropout=0.0,
                    n_out=n_out_feature,
                    layernorm=True,
                    hidden_expansion=2,
                    hidden_layers=1,
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

        if self.embedding_strategy == "binned":
            self.gene_emb = nn.Embedding(self.seq_len+ 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_bin_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=0)
        elif self.embedding_strategy == "continuous":
            n_input = 16
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_emb_input = nn.Embedding(self.seq_len + 1, n_input, padding_idx=self.seq_len)
            self.gene_val_emb = MLPEmbedding(self.seq_dim, n_input=n_input, linear=self.linear_embedding)

        if self.RDA:
            self.depth_val_emb = MLPEmbedding(self.seq_dim, linear=self.linear_embedding)
            self.depth_emb = nn.Parameter(torch.randn(1, 2, self.seq_dim))

        self.target_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)

    def _target_embedding(self, gene_ids: torch.Tensor) -> torch.Tensor:

        # return self.gene_emb(gene_ids)
        return self.target_emb(gene_ids)

    def _gene_embedding(
        self,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        depths: torch.Tensor,
    ) -> torch.Tensor:

        if self.embedding_strategy == "binned":
            output = self.gene_emb(gene_ids) + self.gene_bin_emb(gene_vals.to(torch.long))
        elif self.embedding_strategy == "continuous":
            gene_input = self.gene_val_emb_input(gene_ids)
            output = self.gene_emb(gene_ids) + self.gene_val_emb(gene_vals.unsqueeze(-1) * gene_input)
        elif self.embedding_strategy == "film":
            output = self.gene_emb(gene_ids) + self.gene_scale(gene_ids) * self.gene_val_emb(gene_vals)

        if self.RDA:
            depth_emb =  self.depth_val_emb(depths) + self.depth_emb
            output = torch.concat((output, depth_emb), dim=1)

        return output

    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_target_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        depths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)

        key_vals = self._gene_embedding(gene_ids, gene_vals, depths)
        gene_query = self._target_embedding(gene_target_ids)

        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )

        latent = self.gmlp(latent)
        gene_pred = self.decoder(latent, gene_query)

        if self.cell_properties is not None:
            feature_pred = {}
            for k in self.cell_properties.keys():

                query = self.feature_emb[k].repeat(len(gene_ids), 1, 1).to(latent.device)
                #print("A", k, latent.shape, query.shape)
                feature_pred[k] = self.feature_decoder[k](latent, query)

        else:
            feature_pred = None

        return gene_pred, latent, feature_pred


class Exceiver_atacseq(nn.Module):

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
        n_chr: int = 23,
        dropout: float = 0.0,  # for the process attention module
        n_bins: Optional[int] = None,
        n_emb_per_chr: int = 4000,
        predict_atac: bool = False,
        **kwargs,
    ):

        super().__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.n_bins = n_bins
        self.n_chr = n_chr
        self.n_emb_per_chr = n_emb_per_chr
        self.predict_atac = predict_atac
        # create the gene embeddings based on whether to use rank ordering
        self._create_gene_embeddings()
        self._create_atac_embeddings()

        # self.output_emb = nn.Embedding(seq_len + 1, seq_dim, padding_idx=seq_len)
        self.query_emb = nn.Parameter(torch.randn(query_len, query_dim))

        self.encoder_cross_attn = CrossAttn(
            query_dim, seq_dim, num_heads=n_heads,
        )

        self.process_self_attn = ProcessSelfAttn(
            query_dim, n_layers, n_heads, dim_feedforward, dropout,
        )

        if self.predict_atac:
            self.decoder = ATACDecoder(seq_dim, query_dim, 128, dropout)
        else:
            self.decoder = Decoder(seq_dim, query_dim, dropout)

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

        if self.n_bins is not None:
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_bin_emb = nn.Embedding(self.n_bins + 1, self.seq_dim, padding_idx=0)

        else:
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_w = nn.Embedding(self.seq_len + 1, 1, padding_idx=self.seq_len)
            self.gene_val_b = nn.Embedding(self.seq_len + 1, 1, padding_idx=self.seq_len)

            torch.nn.init.ones_(self.gene_val_w.weight)
            torch.nn.init.constant_(self.gene_val_b.weight, -0.25)

    def _create_atac_embeddings(self):

        #self.chr_emb = nn.Embedding(self.n_chr + 1, self.seq_dim, padding_idx=self.n_chr)
        #self.pos_emb = PosEmbeddingSinuosidal(self.seq_dim)
        #self.pos_emb = RadialBasisEmbeddingChr(
        #    23, self.seq_dim, n_emb_per_chr=self.n_emb_per_chr,
        #)
        #n = self.n_chr * self.n_emb_per_chr + 2
        #self.pos_emb = nn.Embedding(n, self.seq_dim, padding_idx=n - 1)
        self.atac_emb = nn.Parameter(torch.randn(1, 1, self.seq_dim))

        """
        self.conv_layer = nn.Sequential(*[
            nn.Flatten(start_dim=0, end_dim=1),
            nn.Conv1d(1, 4, 13, stride=2),
            nn.ReLU(),
            nn.Conv1d(4, 8, 7, stride=2),
            nn.ReLU(),
            nn.Conv1d(8, 8, 7, stride=2),
            nn.Flatten(start_dim=1),
        ])
        
        n_out = 216

        self.linear =  nn.Sequential(*[
            nn.Linear(self.seq_dim+n_out, self.seq_dim),
            nn.ReLU(),
            nn.Linear(self.seq_dim, self.seq_dim),
        ])
        """
        #self.linear = nn.Linear(256 // 4, 1)

        self.max_pool = nn.Sequential(*[
            nn.Flatten(start_dim=0, end_dim=1),
            nn.MaxPool1d(4, stride=2),
            nn.Flatten(start_dim=1),
        ])

        n = 128 // 2 - 1

        self.W = nn.Parameter(torch.ones(self.seq_len, n))
        self.b = nn.Parameter(torch.zeros(self.seq_len, 1))




    def _target_embedding(self, gene_ids: torch.Tensor) -> torch.Tensor:

        return self.gene_emb(gene_ids)

    def _gene_embedding(self,gene_ids: torch.Tensor,gene_vals: torch.Tensor) -> torch.Tensor:

        if self.n_bins is not None:
            return self.gene_emb(gene_ids) + self.gene_bin_emb(gene_vals.to(torch.long))
        else:
            gene_emb = self.gene_emb(gene_ids)
            vals = gene_vals.unsqueeze(2)  # * (self.gene_val_w(gene_ids) +  self.gene_val_b(gene_ids))
            return vals * gene_emb

    def _atac_embedding(
        self,
        atac_gene_based: torch.Tensor,
        gene_emb: torch.Tensor,
        gene_target_ids: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = atac_gene_based.shape[0]
        # return self.pos_emb(atac_chr, atac_pos)
        # return self.gene_emb(atac_chr) + self.pos_emb(atac_pos)
        #return self.pos_emb(atac_pos_abs)

        #print("D", atac_gene_based.shape)
        #y = self.conv_layer(atac_gene_based)
        # print("A", y.shape)
        #y = y.reshape(batch_size, -1, 216)
        #print("A1", y.shape)
        #print("B", gene_emb.shape)

        y = self.max_pool(atac_gene_based)
        y = y.reshape(batch_size, -1, 128 // 2 - 1)
        #y = self.linear(y)
        #y = self.linear(torch.squeeze(atac_gene_based))

        alpha = torch.sum(self.W[gene_target_ids] * y, dim=-1, keepdim=True) + self.b[gene_target_ids]

        emb = alpha * gene_emb

        #print("C", emb.shape)
        #atac_gene_based = torch.squeeze(atac_gene_based)
        # y = torch.cat((y, emb), dim=-1)
        return emb



    def forward(
        self,
        #atac_chr: torch.Tensor,
        #atac_pos: torch.Tensor,
        #atac_pos_abs: torch.Tensor,
        atac_gene_based: torch.Tensor,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        gene_target_ids: torch.Tensor,
        key_padding_mask_atac: Optional[torch.Tensor] = None,
        key_padding_mask_genes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = len(gene_ids)
        input_query = self.query_emb.repeat(batch_size, 1, 1)

        gene_key_vals = self._gene_embedding(gene_ids, gene_vals)
        gene_query = self._target_embedding(gene_target_ids)

        # atac_key_vals = self._atac_embedding(atac_chr, atac_pos)
        # gene query and atac_gene_based vals come from same genes
        #atac_key_vals = self._atac_embedding(atac_gene_based, gene_query, gene_target_ids)

        #key_padding_mask = torch.cat((key_padding_mask_atac, key_padding_mask_genes), dim=1)
        #key_vals = torch.cat((atac_key_vals, gene_key_vals), dim=1)

        key_padding_mask = key_padding_mask_genes
        key_vals = gene_key_vals


        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )
        latent = self.process_self_attn(latent)

        if self.predict_atac:
            atac_pred = self.decoder(latent, gene_query)
            gene_pred = None
        else:
            gene_pred = self.decoder(latent, gene_query)
            atac_pred = None

        return gene_pred, atac_pred, latent


class RadialBasisEmbeddingChr(nn.Module):

    def __init__(self, n_chromosomes, embedding_dim, n_emb_per_chr: int = 512):
        super().__init__()

        self.embedding_dim = embedding_dim

        dtype = torch.float32

        """
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10_000.0) / embedding_dim))
        div_term = div_term[None, None, :]
        pos_emb = torch.zeros((n_chromosomes, n_emb_per_chr, embedding_dim), dtype=torch.bfloat16)
        pos = torch.arange(n_emb_per_chr)
        pos_emb[:, :, ::2] = torch.cos(pos[None, :, None] * div_term)
        pos_emb[:, :, 1::2] = torch.sin(pos[None, :, None] * div_term)

        chr_emb =  torch.randn(n_chromosomes, 1, embedding_dim, dtype=torch.bfloat16)

        # Trainable embeddings for each chromosome
        self.embeddings = nn.Parameter(
            pos_emb + chr_emb
        )
        """

        self.embeddings = nn.Parameter(
            torch.randn(n_chromosomes, n_emb_per_chr, embedding_dim, dtype=dtype),
        )


        centers = np.linspace(0, 1, n_emb_per_chr)[None, :]
        centers = np.tile(centers, (n_chromosomes, 1))

        # Trainable centers for RBFs per chromosome
        self.centers = nn.Parameter(
            torch.from_numpy(centers)
        ).to(dtype).view(1, 1, n_chromosomes, n_emb_per_chr)

        # Trainable variances (log-transformed for numerical stability)
        self.log_variances = nn.Parameter(
            torch.full((1, 1, n_chromosomes, n_emb_per_chr), -5.0).to(dtype)
        )

    def forward(self, chromosome, position):

        weighted_embedding = []
        batch_size = position.shape[0]

        #print("SS", self.log_variances.shape)
        #print("SSSS", self.log_variances[:, 0, :].shape)

        for ch in range(23):

            pos = position[:, ch, :].unsqueeze(-1)
            variances = torch.exp(self.log_variances[:, :, ch, :]).to(position.device)
            centers = self.centers[:, :, ch, :].to(position.device)

            #print("A", pos.shape, centers.shape, variances.shape, centers.shape)
            weights = torch.exp(-((pos - centers) ** 2) / (2 * variances))
            weights = F.softmax(weights, dim=-1)
            #print("B", weights.shape, self.embeddings[ch, ...].shape)

            weighted_embedding.append(weights @ self.embeddings[ch, ...])

        weighted_embedding = torch.stack(weighted_embedding, 1).reshape(batch_size, -1, self.embedding_dim )
        #print("weighted_embedding", weighted_embedding.shape)

        return weighted_embedding

class RadialBasisEmbedding(nn.Module):

    def __init__(self, n_chromosomes, embedding_dim, n_emb_per_chr: int = 512, chr_jump: float = 2.0):
        super().__init__()

        self.chr_jump = chr_jump
        # Trainable embeddings for each chromosome
        self.embeddings = nn.Parameter(
            torch.randn(n_chromosomes * n_emb_per_chr, embedding_dim, dtype=torch.bfloat16),
        )

        centers = []
        for n in range(n_chromosomes):
            s = np.linspace(0, 1, n_emb_per_chr) + n * chr_jump
            centers += s.tolist()
        centers = np.array(centers)

        # Trainable centers for RBFs per chromosome
        self.centers = nn.Parameter(torch.from_numpy(centers)).to(torch.bfloat16).view(1, 1, -1)

        # Trainable variances (log-transformed for numerical stability)
        self.log_variances = nn.Parameter(
            torch.full((n_chromosomes * n_emb_per_chr, 1), -5.0).to(torch.bfloat16)
        ).view(1, 1, -1)

    def forward(self, chromosome, position):

        position = position + chromosome * self.chr_jump
        position = position[..., None]
        variances = torch.exp(self.log_variances).to(position.device)
        self.centers = self.centers.to(position.device)

        weights = torch.exp(-((position - self.centers) ** 2) / (2 * variances))
        weight_topk,  indices = torch.topk(weights, k=10, dim=-1)
        weight_topk = weight_topk / weight_topk.sum(dim=-1, keepdim=True)

        print("A", self.embeddings.shape, indices.shape)
        embeddings = torch.gather(self.embeddings, -1, indices)

        #weighted_embedding = (weights * self.embeddings).sum(dim=0)
        #weighted_embedding = weights @ self.embeddings
        weighted_embedding = weight_topk @ embeddings

        return weighted_embedding


class ATACSeqEmbedding(nn.Module):
    def __init__(self, num_chromosomes, embedding_dim, n_emb_per_chr: int = 512):
        super().__init__()
        self.num_chromosomes = num_chromosomes
        self.n_emb_per_chr = n_emb_per_chr
        self.embedding_dim = embedding_dim

        # Trainable embeddings for each chromosome
        self.embeddings = nn.ParameterDict({
            str(chr_idx): nn.Parameter(torch.randn(n_emb_per_chr, embedding_dim))
            for chr_idx in range(0, num_chromosomes)
        })

        # Trainable centers for RBFs per chromosome
        self.centers = nn.ParameterDict({
            str(chr_idx): nn.Parameter(torch.linspace(0, 1, n_emb_per_chr).unsqueeze(-1))
            for chr_idx in range(0, num_chromosomes)
        })

        # Trainable variances (log-transformed for numerical stability)
        self.log_variances = nn.ParameterDict({
            str(chr_idx): nn.Parameter(torch.full((n_emb_per_chr, 1), -2.0))
            for chr_idx in range(0, num_chromosomes)
        })

    def forward(self, chromosome, position):
        """
        Args:
            chromosome: Tensor of shape (batch_size,) containing chromosome indices (1-based).
            position: Tensor of shape (batch_size, 1) containing positions (normalized between 0 and 1).

        Returns:
            Tensor of shape (batch_size, embedding_dim) containing the weighted sum of embeddings.
        """
        batch_size = position.shape[0]
        output_embeddings = torch.zeros(batch_size, self.embedding_dim, device=position.device)

        for i in range(batch_size):
            #chr_idx = str(chromosome[i].item())
            chr_idx = [str(int(c)) for c in chromosome[i].tolist()]
            pos = position[i].unsqueeze(0)  # Shape (1,1)

            # Retrieve trainable embeddings, centers, and variances
            centers = self.centers[chr_idx]  # (n_emb_per_chr, 1)
            variances = torch.exp(self.log_variances[chr_idx])  # (n_emb_per_chr, 1), ensure positivity

            # Compute RBF weights
            weights = torch.exp(-((pos - centers) ** 2) / (2 * variances))  # (n_emb_per_chr, 1)
            weights = weights / weights.sum()  # Normalize weights


            # Compute weighted sum
            weighted_embedding = (weights * self.embeddings).sum(dim=0)  # (embedding_dim,)
            output_embeddings[i] = weighted_embedding

        return output_embeddings


# Example usage


class PosEmbeddingSinuosidal(nn.Module):

    def __init__(self, d_model, scale_factor: float = 5000.0):

        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor # position inputs range from 0 to 1, multiply by scale factor work well
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(5_000.0) / d_model))
        self.div_term = self.div_term[None, None, :]

    def forward(self, pos):

        y = torch.zeros((*pos.shape, self.d_model), dtype=torch.float32).to(pos.device)
        pos = pos * self.scale_factor
        y[:, :, ::2] = torch.cos(pos[:, :, None] * self.div_term.to(pos.device))
        y[:, :, 1::2] = torch.sin(pos[:, :, None] * self.div_term.to(pos.device))
        return y

class PosEmbedding(nn.Module):

    def __init__(self, d_model, max_len=13941):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len + 1, d_model, padding_idx=max_len)

    def forward(self, pos):

        #pos0 = torch.clip(pos-2, 0, self.max_len)
        pos1 = torch.clip(pos-1, 0, self.max_len)
        pos2 = torch.clip(pos, 0, self.max_len)
        pos3 = torch.clip(pos+1, 0, self.max_len)
        #pos4 = torch.clip(pos+2, 0, self.max_len)

        y =  0.5 * self.emb(pos1) + self.emb(pos2) + 0.5 * self.emb(pos3)

        return y




def load_model(model_save_path, model):

    params_loaded = []
    non_network_params = []
    state_dict = {}
    ckpt = torch.load(model_save_path)
    key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    for k, v in ckpt[key].items():
        loaded = False
        if "cell_property" in k:
            non_network_params.append(k)
        elif "network" in k:
            k = k.split(".")
            k = ".".join(k[1:])
        for n, p in model.named_parameters():
            if n == k and p.size() == v.size():
                state_dict[k] = v
                params_loaded.append(n)
                loaded = True
        if not loaded:
            print(f"{k} not loaded")

    model.load_state_dict(state_dict, strict=False)
    print(f"Number of params loaded: {len(params_loaded)}")
    print(f"Non-network parameters not loaded: {non_network_params}")

    return model
