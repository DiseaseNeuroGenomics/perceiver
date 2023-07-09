# Code heavily borrowed from https://github.com/keiserlab/exceiver/tree/main/exceiver
# From paper https://arxiv.org/abs/2210.14330
# other possible data from: https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219
# other public covid data: https://onlinelibrary.wiley.com/doi/10.1002/ctd2.104

from typing import Any, Dict, Optional, Tuple
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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            n_heads,
            dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # NYM June 24
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
        cell_properties: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,  # for the process attention module
        rank_order: bool = False,
        bin_gene_count: bool = True,
        use_class_emb: bool = True,
        n_gene_bins: int = 16,
        n_classes: int = 8,
        **kwargs,
    ):

        super().__init__()

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        # rank ordering/binning gene count will change how genes are embedded
        self.rank_order = rank_order
        self.bin_gene_count = bin_gene_count
        self.use_class_emb = use_class_emb
        self.n_classes = n_classes
        self.n_gene_bins = n_gene_bins
        # create the gene embeddings based on whether to use rank ordering
        self._create_gene_embeddings()

        # Embeddings and attention blocks
        self.n_cell_props = len(cell_properties) if cell_properties is not None else 0

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

        print("BIN GENES, and class emb", self.bin_gene_count, self.use_class_emb)

        self.cell_properties = cell_properties
        if self.n_cell_props > 0:
            self.cell_prop_emb = nn.Embedding(self.n_cell_props + 1, seq_dim, padding_idx=self.n_cell_props)
            self.cell_prop_mlp = nn.ModuleDict()
            for k, cell_prop in cell_properties.items():
                # the output size of the cell property prediction MLP will be 1 if the property is continuous;
                # if it is discrete, then it will be the length of the possible values
                n_targets = 1 if not cell_prop["discrete"] else len(cell_prop["values"])
                self.cell_prop_mlp[k] = nn.Sequential(nn.LayerNorm(seq_dim), nn.Linear(seq_dim, n_targets))

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

        self.class_emb = nn.Embedding(self.n_classes + 1, self.seq_dim, padding_idx=self.n_classes)

        if self.bin_gene_count:
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_bin_emb = nn.Embedding(self.n_gene_bins + 1, self.seq_dim, padding_idx=self.n_gene_bins)

        elif self.rank_order:
            self.gene_emb_low = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_emb_high = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            # TODO: I think this initialization below works...need to confirm
            self.gene_emb_low.weight.data = 0.1 * self.gene_emb_low.weight.data + 0.9 * self.gene_emb_high.weight.data
        else:
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_w = nn.Parameter(torch.ones(1, self.seq_len))
            self.gene_val_b = nn.Parameter(torch.zeros(1, self.seq_len))

    def _target_embedding(self, gene_ids: torch.Tensor) -> torch.Tensor:

        return self.gene_emb(gene_ids)

    def _gene_embedding(
        self,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        cell_class_id: torch.Tensor,
    ) -> torch.Tensor:


        if self.bin_gene_count:
            if self.use_class_emb:
                return (self.gene_emb(gene_ids) +
                        self.gene_bin_emb(gene_vals.to(torch.long)) +
                        self.class_emb(cell_class_id.to(torch.long))[:, None, :]
                        ) / 3
            else:
                return (self.gene_emb(gene_ids) + self.gene_bin_emb(gene_vals.to(torch.long))) / 2
        elif self.rank_order:
            return (1 - gene_vals[..., None]) * self.gene_emb_low(gene_ids) + gene_vals[..., None] * self.gene_emb_high(gene_ids)

        else:
            class_emb = self.class_emb(cell_class_id.to(torch.long)).unsqueeze(1) if self.use_class_emb else 0.0
            gene_emb = self.gene_emb(gene_ids)
            return gene_vals.unsqueeze(2) * gene_emb + class_emb



    def forward(
        self,
        gene_ids: torch.Tensor,
        gene_target_ids: torch.Tensor,
        cell_prop_target_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        cell_class_id: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_query = self.query_emb.repeat(len(gene_ids), 1, 1)

        cell_prop_target_query = self.cell_prop_emb(cell_prop_target_ids)

        key_vals = self._gene_embedding(gene_ids, gene_vals, cell_class_id)
        gene_target_query = self._target_embedding(gene_target_ids)

        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )
        latent = self.process_self_attn(latent)

        # Query genes and cell properties
        # Decoder out will contain the latent for both genes and cell properties, concatenated together
        decoder_out, decoder_weights = self.decoder_cross_attn(
            torch.cat((gene_target_query, cell_prop_target_query), dim=1),
            latent,
            key_padding_mask=None,
        )

        # Predict genes
        n_genes = gene_target_query.size(1)
        gene_pred = self.gene_mlp(decoder_out[:, : n_genes, :])

        # Predict cell properties
        if self.n_cell_props > 0:
            cell_prop_pred = {}
            for n, k in enumerate(self.cell_prop_mlp.keys()):
                cell_prop_pred[k] = torch.squeeze(self.cell_prop_mlp[k](decoder_out[:, n_genes + n, :]))

        else:
            cell_prop_pred = None

        return gene_pred, cell_prop_pred, latent


def load_model(model_save_path, model):

    params_loaded = []
    non_network_params = []
    state_dict = {}
    ckpt = torch.load(model_save_path)
    key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    for k, v in ckpt[key].items():

        if "cell_property" in k:
            non_network_params.append(k)
        elif "network" in k:
            k = k.split(".")
            k = ".".join(k[1:])
        for n, p in model.named_parameters():
            if n == k and p.size() == v.size():
                state_dict[k] = v
                params_loaded.append(n)

    model.load_state_dict(state_dict, strict=True)
    print(f"Number of params loaded: {len(params_loaded)}")
    print(f"Non-network parameters not loaded: {non_network_params}")

    return model
