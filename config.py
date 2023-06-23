
dataset_cfg = {
    "data_path": "/home/masse/work/perceiver/data_scaled",
    "predict_classes": {"CERAD": 4, "BRAAK_AD": 7, "class": 8},
    "batch_size": 16,
    "num_workers": 16,
    "n_mask": 316,
    "rank_order": True,
}

trainer_cfg = {
    "accumulate_grad_batches": 4,
    "precision": "bf16",
    "grad_clip_value": 0.5,
}

task_cfg = {
    # "classify": True,
    "learning_rate": 4e-5,
    "weight_decay": 0.00001,
    "warmup_steps": 8_000,
    "decay_steps": 100_000,
    "decay": 0.999985,

}

model_cfg = {
    "seq_dim": 512,
    "query_len": 9,
    "query_dim": 512,
    "n_layers": 12,
    "dim_feedforward": 1024,
    "n_heads": 2,
    "dropout": 0.0,
    "rank_order": dataset_cfg["rank_order"]
}
