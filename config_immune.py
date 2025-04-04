
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""

cell_properties = None

embedding_strategy = "continuous"
n_bins = None

dataset_cfg = {
    "data_path": "/home/masse/work/GRN/data/data_psych_ad.dat",
    "metadata_path": "/home/masse/work/GRN/data/metadata_psych_ad.pkl",
    "cell_properties": cell_properties,
    "batch_size": 64,
    "num_workers": 10,
    "n_mask": 3_000,
    "n_input": 4_000,
    "remove_sex_chrom": False,
    "protein_coding_only": False,
    "rank_order": False,
    "cell_prop_same_ids": False,
    "embedding_strategy": embedding_strategy,
    "n_bins": n_bins,
}


trainer_cfg = {
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
    "grad_clip_value": 1.0,
    "n_devices": 1,
}

task_cfg = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-7,
    "warmup_steps": 25_000,
    "decay_steps": 150_000,
    "decay": 0.999985,
    "balance_classes": True,
    "gene_weight": 1.0,
    "cell_prop_weight": 0.0,
}

model_cfg = {
    "seq_dim": 512,
    "query_len": 96,
    "query_dim": 512,
    "n_layers": 20,
    "n_heads": 8,
    "dim_feedforward": 1024,
    "dropout": 0.0,
    "embedding_strategy": embedding_strategy,
    "n_bins": n_bins,
}

test_cfg = {
    #"ckpt_path": "/home/masse/work/perceiver/mssm_model/epoch39_v195.ckpt",
    "ckpt_path": "/home/masse/work/perceiver/saved_model/gmlp16_gene_expression.ckpt"
}

test_dataset_cfg = {
    "train_data_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/test_data.dat",
    "train_metadata_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/test_metadata.pkl",
    "test_data_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/test_data.dat",
    "test_metadata_path": "/home/masse/work/perceiver/mssm_rush_data_all_genes/test_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 16,
    "num_workers": 10,
    "n_mask": 200,
    "rank_order": False,
    "cell_prop_same_ids": False,
    "bin_gene_count": False,
}
