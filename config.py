
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""
cell_properties = {
    "AD": {"discrete": True, "values": [0, 1]},
    "Dementia": {"discrete": True, "values": [0, 1]},
    "CERAD": {"discrete": True, "values": [1,2,3,4]},
    "BRAAK_AD": {"discrete": True, "values": [0,1,2,3,4,5,6]},
    #"ApoE_gt": {"discrete": True, "values": [23, 24, 33, 34, 44]},
    #"Sex": {"discrete": True, "values": ["Male", "Female"]},
    #"class": {"discrete": True, "values": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo']},
    #"Age": {"discrete": False, "values": [-1]},
    #"PMI": {"discrete": False},
    #"SubID": {"discrete": True, "values": None},
}


dataset_cfg = {
    #"train_data_path": "/ssd/mssm_rush_data/train_data.dat",
    #"train_metadata_path": "/ssd/mssm_rush_data/train_metadata.pkl",
    #"test_data_path": "/ssd/mssm_rush_data/test_data.dat",
    #"test_metadata_path": "/ssd/mssm_rush_data/test_metadata.pkl",
    "train_data_path": "/home/masse/work/perceiver/mssm_rush_data/train_data.dat",
    "train_metadata_path": "/home/masse/work/perceiver/mssm_rush_data/train_metadata.pkl",
    "test_data_path": "/home/masse/work/perceiver/mssm_rush_data/test_data.dat",
    "test_metadata_path": "/home/masse/work/perceiver/mssm_rush_data/test_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 128,
    "num_workers": 10,
    "n_mask": 1,
    "rank_order": False,
    "cell_prop_same_ids": False,
    "cutmix_pct": 0.0,
    "mixup": False,
    "bin_gene_count": False,
}

dataset_memory_cfg = {
    "adata_path": "/home/masse/work/perceiver/h5ad/RUSH_2023-06-08_21_44.h5ad",
    "cell_properties": cell_properties,
    "batch_size": 128,
    "num_workers": 1,
    "n_mask": 100,
    "rank_order": False,
    "cell_prop_same_ids": True,
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
    "learning_rate": 1.0e-4,
    "weight_decay": 1e-6,
    "warmup_steps": 10_000,
    "decay_steps": 150_000,
    "decay": 0.999988,
    "balance_classes": True,
    "gene_weight": 0.0,
    "cell_prop_weight": 1.0,
    "variational": True,
    "bottleneck_dim": 32,
}

model_cfg = {
    "seq_dim": 256,
    "query_len": 64,
    "query_dim": 256,
    "n_layers": 20,
    "n_heads": 4,
    "dim_feedforward": 1024,
    "dropout": 0.0,
    "n_gene_bins": 16,
    "variational": True,
    "bottleneck_dim": 32,
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
