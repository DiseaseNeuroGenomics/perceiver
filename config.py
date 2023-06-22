
dataset_cfg = {
    "data_path": "/home/masse/work/perceiver/data",
    "predict_classes": {"CERAD": 4, "BRAAK_AD": 7, "class": 8},
    "batch_size": 32,
    "num_workers": 16,
    "n_mask": 316,
    "rank_order": True,
}

trainer_cfg = {
    "accumulate_grad_batches": 2,
    "precision": "bf16",
    "grad_clip_value": 0.5,
}

task_cfg = {
    # "classify": True,
    "learning_rate": 5e-5,
    "weight_decay": 0.00001,
    "warmup_steps": 5_000,
    "decay_steps": 100_000,
    "decay": 0.999985,

}

model_cfg = {
    "seq_dim": 256,
    "query_len": 64,
    "query_dim": 256,
    "n_layers": 10,
    "dim_feedforward": 1024,
    "n_heads": 2,
    "dropout": 0.0,
    "rank_oder": dataset_cfg["rank_order"]
}
