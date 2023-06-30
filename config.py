
dataset_cfg = {
    "train_data_path": "/ssd/mssm_data/train_data.dat",
    "train_metadata_path": "/ssd/mssm_data/train_metadata.pkl",
    "test_data_path": "/ssd/rush_data/train_data.dat",
    "test_metadata_path": "/ssd/rush_data/train_metadata.pkl",
    "predict_classes": ["BRAAK_AD", "CERAD", "class"],
    "batch_size": 128,
    "num_workers": 16,
    "n_mask": 100,
    "rank_order": True,
}

trainer_cfg = {
    "accumulate_grad_batches": 1,
    "precision": "bf16",
    "grad_clip_value": 0.5,
    "n_devices": 1,
}

task_cfg = {
    # "classify": True,
    "learning_rate": 8e-5,
    "weight_decay": 0.00001,
    "warmup_steps": 8_000,
    "decay_steps": 100_000,
    "decay": 0.999985,

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
