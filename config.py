
dataset_cfg = {
    "data_path": "/home/masse/work/mssm/perceiver/data",
    "predict_classes": {"CERAD": 4, "BRAAK_AD": 7, "class": 8},
    "batch_size": 16,
    "num_workers": 16,
    "n_mask": 316,
}

trainer_cfg = {
    "accumulate_grad_batches": 2,
    "precision": "bf16",
    "grad_clip_value": 0.5,
}

task_cfg = {
    # "classify": True,
    "learning_rate": 4e-5,
    "weight_decay": 0.0001,
    "warmup_steps": 5_000,
    "decay_steps": 100_000,
    "decay": 0.999985,

}

model_cfg = {
    "seq_dim": 512,
    "query_len": 64,
    "query_dim": 512,
    "n_layers": 12,
    "dim_feedforward": 1024,
    "n_heads": 4,
    "dropout": 0.0,
}
