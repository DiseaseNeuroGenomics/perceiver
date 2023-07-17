# Code heavily borrowed from https://github.com/keiserlab/exceiver/tree/main/exceiver
# From paper https://arxiv.org/abs/2210.14330
# other possible data from: https://figshare.com/articles/dataset/Tabula_Sapiens_release_1_0/14267219
# other public covid data: https://onlinelibrary.wiley.com/doi/10.1002/ctd2.104

from typing import Any, Dict, Optional, Tuple
import torch
from torch import nn
from functional import GradReverseLayer


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

class CrossAttn(nn.Module):
    def __init__(self, query_dim: int, key_val_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layernorm_kv = nn.LayerNorm(key_val_dim)
        self.layernorm_q = nn.LayerNorm(query_dim)
        print("CrossAttn drop", dropout)
        self.cross_attn = nn.MultiheadAttention(
            query_dim,
            num_heads,
            dropout=dropout,
            kdim=key_val_dim,
            vdim=key_val_dim,
            batch_first=True,
        )
        self.mlp = MLP(query_dim, query_dim)

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
        print("ProcessSelfAttn drop", dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            n_heads,
            dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # NYM June 24
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
        bin_gene_count: bool = False,
        use_class_emb: bool = False,
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
        self.cell_emb = nn.Embedding(self.n_cell_props + 1, seq_dim, padding_idx=self.n_cell_props)

        self.encoder_cross_attn = CrossAttn(
            query_dim, seq_dim, num_heads=n_heads,
        )

        self.process_self_attn = ProcessSelfAttn(
            query_dim, n_layers, n_heads, dim_feedforward, dropout,
        )

        self.decoder = Decoder(seq_dim, query_dim, cell_properties, dropout)

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
            self.gene_bin_emb = nn.Embedding(self.n_gene_bins + 1, self.seq_dim, padding_idx=0)

        elif self.rank_order:
            self.gene_emb_low = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_emb_high = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            # TODO: I think this initialization below works...need to confirm
            self.gene_emb_low.weight.data = 0.1 * self.gene_emb_low.weight.data + 0.9 * self.gene_emb_high.weight.data
        else:
            self.gene_emb = nn.Embedding(self.seq_len + 1, self.seq_dim, padding_idx=self.seq_len)
            self.gene_val_w = nn.Embedding(self.seq_len + 1, 1, padding_idx=self.seq_len)
            self.gene_val_b = nn.Embedding(self.seq_len + 1, 1, padding_idx=self.seq_len)

            torch.nn.init.ones_(self.gene_val_w.weight)
            torch.nn.init.zeros_(self.gene_val_b.weight)


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
            vals = gene_vals.unsqueeze(2) * self.gene_val_w(gene_ids) +  self.gene_val_b(gene_ids)
            return vals * gene_emb + class_emb



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

        cell_query = self.cell_emb(cell_prop_target_ids)

        key_vals = self._gene_embedding(gene_ids, gene_vals, cell_class_id)
        gene_query = self._target_embedding(gene_target_ids)

        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )
        latent = self.process_self_attn(latent)

        gene_pred, cell_pred = self.decoder(latent, gene_query, cell_query)

        return gene_pred, cell_pred, latent


class gMLP_stack(nn.Module):
    def __init__(self, seq_len: int, seq_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        self.gmlps = nn.Sequential(
            *[gMLP(seq_len, seq_dim, hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.gmlps(input)

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
        torch.nn.init.xavier_normal_(self.linear1.weight, gain=0.05)



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


class Decoder(nn.Module):

    def __init__(
        self,
        seq_dim: int,
        query_dim: int,
        cell_properties: Dict[str, Any],
        dropout: float = 0.0,  # for the process attention module
    ):

        super().__init__()

        self.decoder_cross_attn = CrossAttn(
            seq_dim, query_dim
        )  # query is now gene embedding

        self.gene_mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_dim, 1),
        )

        self.cell_mlp = nn.ModuleDict()
        for k, cell_prop in cell_properties.items():
            # the output size of the cell property prediction MLP will be 1 if the property is continuous;
            # if it is discrete, then it will be the length of the possible values
            n_targets = 1 if not cell_prop["discrete"] else len(cell_prop["values"])

            self.cell_mlp[k] = nn.Sequential(
                nn.LayerNorm(seq_dim),
                nn.Linear(seq_dim, seq_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(seq_dim, n_targets),
            )

    def forward(self, latent: torch.Tensor, gene_query: torch.Tensor, cell_query: torch.Tensor):

        # Query genes and cell properties
        # Decoder out will contain the latent for both genes and cell properties, concatenated together
        decoder_out, _ = self.decoder_cross_attn(
            torch.cat((gene_query, cell_query), dim=1),
            latent,
            key_padding_mask=None,
        )

        # Predict genes
        n_genes = gene_query.size(1)
        gene_pred = self.gene_mlp(decoder_out[:, : n_genes, :])

        # Predict cell properties
        cell_pred = {}
        for n, k in enumerate(self.cell_mlp.keys()):
            cell_pred[k] = torch.squeeze(self.cell_mlp[k](decoder_out[:, n_genes + n, :]))

        return gene_pred, cell_pred


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

        self.decoder = Decoder(seq_dim, query_dim, cell_properties, dropout)

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


    def _target_embedding(self, gene_ids: torch.Tensor) -> torch.Tensor:

        return self.gene_emb(gene_ids)

    def _gene_embedding(
        self,
        gene_ids: torch.Tensor,
        gene_vals: torch.Tensor,
        cell_class_id: torch.Tensor,
    ) -> torch.Tensor:

        gene_emb = self.gene_emb(gene_ids)
        vals = gene_vals.unsqueeze(2) * self.gene_val_w(gene_ids) +  self.gene_val_b(gene_ids)
        return vals * gene_emb



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

        cell_query = self.cell_emb(cell_prop_target_ids)

        key_vals = self._gene_embedding(gene_ids, gene_vals, cell_class_id)
        gene_query = self._target_embedding(gene_target_ids)

        # Main calculation of latent variables
        latent, encoder_weights = self.encoder_attn_step(
            key_vals, input_query, key_padding_mask
        )

        latent = self.gmlp(latent)

        gene_pred, cell_pred = self.decoder(latent, gene_query, cell_query)

        return gene_pred, cell_pred, latent


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
