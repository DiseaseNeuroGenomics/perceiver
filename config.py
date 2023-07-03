
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""
cell_properties = {
    "AD": {"discrete": True, "values": [0, 1]},
    "Dementia": {"discrete": True, "values": [0, 1]},
    "CERAD": {"discrete": True, "values": [1, 2, 3, 4]},
    "BRAAK_AD": {"discrete": True, "values": [0, 1, 2, 3, 4, 5, 6]},
    "ApoE_gt": {"discrete": True, "values": [23, 24, 33, 34, 44]},
    "class": {"discrete": True, "values": ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo']},
    #"Age": {"discrete": False},
    #"PMI": {"discrete": False},
    "SubID": {"discrete": True, "values": None},
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
    #"train_data_path": "/home/masse/work/perceiver/rush_raw_data/test_data.dat",
    #"train_metadata_path": "/home/masse/work/perceiver/rush_raw_data/test_metadata.pkl",
    #"test_data_path": "/home/masse/work/perceiver/rush_raw_data/test_data.dat",
    #"test_metadata_path": "/home/masse/work/perceiver/rush_raw_data/test_metadata.pkl",
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
    "learning_rate": 0.01,
    "weight_decay": 0.00001,
    "warmup_steps": 12_000,
    "decay_steps": 100_000,
    "decay": 0.999988,
    "balance_classes": False,

}

model_cfg = {
    "seq_dim": 256,
    "query_len": 64,
    "query_dim": 256,
    "n_layers": 8,
    "dim_feedforward": 1024,
    "n_heads": 2,
    "dropout": 0.0,
    "rank_order": dataset_cfg["rank_order"]
}

test_cfg = {
    #"ckpt_path": "/home/masse/work/perceiver/mssm_model/epoch19_v52.ckpt",
    "ckpt_path": "/home/masse/work/perceiver/lightning_logs/version_13/saved_model.ckpt"
}

test_dataset_cfg = {
    "train_data_path": "/home/masse/work/perceiver/rush_raw_data/test_data.dat",
    "train_metadata_path": "/home/masse/work/perceiver/rush_raw_data/test_metadata.pkl",
    "test_data_path": "/home/masse/work/perceiver/rush_raw_data/test_data.dat",
    "test_metadata_path": "/home/masse/work/perceiver/rush_raw_data/test_metadata.pkl",
    "cell_properties": cell_properties,
    "batch_size": 16,
    "num_workers": 10,
    "n_mask": 200,
    "rank_order": False,
}
