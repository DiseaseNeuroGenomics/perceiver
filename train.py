
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver
from tasks import MSELoss, AdverserialLoss
from config import dataset_cfg, task_cfg, model_cfg, trainer_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')


def main(adverserial: bool=True):

    # Set seed
    pl.seed_everything(2299)

    # Set up data module
    dm = DataModule(**dataset_cfg)

    dm.setup(None)

    # Transfer information from Dataset
    model_cfg["seq_len"] = dm.train_dataset.n_genes
    model_cfg["cell_properties"] = dm.train_dataset.cell_properties
    task_cfg["cell_properties"] = dm.train_dataset.cell_properties

    # Create network
    model = Exceiver(**model_cfg)

    if adverserial:
        task = AdverserialLoss(network=model, task_cfg=task_cfg)
    else:
        task = MSELoss(network=model, task_cfg=task_cfg)


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=1000,
        gradient_clip_val=trainer_cfg["grad_clip_value"] if not adverserial else None,
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=400,
        limit_val_batches=400,

    )

    trainer.fit(task, dm)



if __name__ == "__main__":

    main()