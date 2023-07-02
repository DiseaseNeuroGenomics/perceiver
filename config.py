
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""
cell_properties = {
    "AD": [0, 1],
    "Dementia": [0, 1],
    "CERAD": [1, 2, 3, 4],
    "BRAAK_AD": [0, 1, 2, 3, 4, 5, 6],
    "ApoE_gt": [23, 24, 33, 34, 44],
    "Age": None,
    "PMI": None,
    "class": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'],
}


dataset_cfg = {
    "train_data_path": "/ssd/mssm_raw_data/train_data.dat",
    "train_metadata_path": "/ssd/mssm_raw_data/train_metadata.pkl",
    "test_data_path": "/ssd/mssm_raw_data/test_data.dat",
    "test_metadata_path": "/ssd/mssm_raw_data/test_metadata.pkl",
    #"train_data_path": "/sc/arion/projects/psychAD/massen06/mssm_raw_data/train_data.dat",
    #"train_metadata_path": "/sc/arion/projects/psychAD/massen06/mssm_raw_data/train_metadata.pkl",
    #"test_data_path": "/sc/arion/projects/psychAD/massen06/rush_raw_data/train_data.dat",
    #"test_metadata_path": "/sc/arion/projects/psychAD/massen06/rush_raw_data/train_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 64,
    "num_workers": 10,
    "n_mask": 200,
    "rank_order": False,
}

trainer_cfg = {
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
    "grad_clip_value": 0.5,
    "n_devices": 2,
}

task_cfg = {
    # "classify": True,
    "learning_rate": 12e-5,
    "weight_decay": 0.00001,
    "warmup_steps": 12_000,
    "decay_steps": 100_000,
    "decay": 0.999988,
    "balance_classes": False,

}

model_cfg = {
    "seq_dim": 512,
    "query_len": 64,
    "query_dim": 512,
    "n_layers": 10,
    "dim_feedforward": 1024,
    "n_heads": 2,
    "dropout": 0.0,
    "rank_order": dataset_cfg["rank_order"]
}
