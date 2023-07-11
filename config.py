
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""
cell_properties = {
    "AD": {"discrete": True, "values": [0, 1]},
    "Dementia": {"discrete": True, "values": [0, 1]},
    "CERAD": {"discrete": False, "values": [-1]},
    "BRAAK_AD": {"discrete": False, "values": [-1]},
    #"ApoE_gt": {"discrete": True, "values": [23, 24, 33, 34, 44]},
    #"class": {"discrete": True, "values": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo']},
    #"Age": {"discrete": False},
    #"PMI": {"discrete": False},
    # "SubID": {"discrete": True, "values": None},
}


dataset_cfg = {
    #"train_data_path": "/ssd/mssm_rush_data/train_data.dat",
    #"train_metadata_path": "/ssd/mssm_rush_data/train_metadata.pkl",
    #"test_data_path": "/ssd/mssm_rush_data/test_data.dat",
    #"test_metadata_path": "/ssd/mssm_rush_data/test_metadata.pkl",
    "train_data_path": "/home/masse/work/perceiver/rush_raw_data/train_data.dat",
    "train_metadata_path": "/home/masse/work/perceiver/rush_raw_data/train_metadata.pkl",
    "test_data_path": "/home/masse/work/perceiver/rush_raw_data/test_data.dat",
    "test_metadata_path": "/home/masse/work/perceiver/rush_raw_data/test_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 256,
    "num_workers": 10,
    "n_mask": 200,
    "rank_order": False,
    "cell_prop_same_ids": False,
    "cutmix_pct": 0.0,
    "mixup": False,
    "bin_gene_count": False,
}

dataset_memory_cfg = {
    "adata_path": "/home/masse/work/perceiver/h5ad/RUSH_2023-06-08_21_44.h5ad",
    "cell_properties": cell_properties,
    "batch_size": 256,
    "num_workers": 1,
    "n_mask": 200,
    "rank_order": False,
    "cell_prop_same_ids": False,
    "cutmix_pct": 0.0,
    "mixup": False,
    "bin_gene_count": False,
}

trainer_cfg = {
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
    "grad_clip_value": 0.5,
    "n_devices": 1,
}

task_cfg = {
    # "classify": True,
    "learning_rate": 1.2e-4,
    "weight_decay": 1e-7,
    "warmup_steps": 12_000,
    "decay_steps": 100_000,
    "decay": 0.999988,
    "balance_classes": True,
    "adverserial": False,
    "gene_weight": 0.5,
    "cell_prop_weight": 1.0,
}

model_cfg = {
    "seq_dim": 64,
    "query_len": 64,
    "query_dim": 64,
    "n_layers": 4,
    "dim_feedforward": 512,
    "n_heads": 4,
    "dropout": 0.0,
    "n_gene_bins": 16,
}

test_cfg = {
    "ckpt_path": "/home/masse/work/perceiver/mssm_model/epoch39_v195.ckpt",
    #"ckpt_path": "/home/masse/work/perceiver/lightning_logs/version_13/saved_model.ckpt"
}

test_dataset_cfg = {
    "train_data_path": "/home/masse/work/perceiver/mssm_rush_data/train_data.dat",
    "train_metadata_path": "/home/masse/work/perceiver/mssm_rush_data/train_metadata.pkl",
    "test_data_path": "/home/masse/work/perceiver/mssm_rush_data/test_data.dat",
    "test_metadata_path": "/home/masse/work/perceiver/mssm_rush_data/test_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 16,
    "num_workers": 10,
    "n_mask": 100,
    "rank_order": False,
    "cell_prop_same_ids": False,
    "bin_gene_count": False,
}
