
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, load_model
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

    # Transfer information from Dataset
    model_cfg["seq_len"] = dm.train_dataset.n_genes
    model_cfg["cell_properties"] = dm.train_dataset.cell_properties
    task_cfg["cell_properties"] = dm.train_dataset.cell_properties

    # Create network
    model = Exceiver(**model_cfg)

    #model = torch.load(test_cfg["ckpt_path"])
    model = load_model(test_cfg["ckpt_path"], model)

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
        limit_val_batches=200,

    )

    trainer.fit(task, dm)



if __name__ == "__main__":

    main()