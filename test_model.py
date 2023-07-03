
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, extract_state_dict
from tasks import MSELoss
from config import test_dataset_cfg, task_cfg, model_cfg, trainer_cfg, test_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')


def main():

    # Set seed
    pl.seed_everything(2299)

    # Set up data module
    dm = DataModule(**test_dataset_cfg)

    dm.setup(None)

    # Create network
    model_cfg["seq_len"] = dm.train_dataset.n_genes
    model_cfg["cell_properties"] = test_dataset_cfg["cell_properties"]

    model = Exceiver(**model_cfg)

    #model = torch.load(test_cfg["ckpt_path"])
    state_dict = extract_state_dict(test_cfg["ckpt_path"])
    model.load_state_dict(state_dict)

    task_cfg["cell_properties"] = test_dataset_cfg["cell_properties"]
    task = MSELoss(
        network=model,
        task_cfg=task_cfg,
    )

    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=1,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=1,
        limit_val_batches=1000,

    )

    trainer.fit(task, dm)



if __name__ == "__main__":

    main()