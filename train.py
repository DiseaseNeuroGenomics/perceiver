import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from datasets import DataModule
from networks import Exceiver, extract_state_dict
from tasks import MSELoss
from config import dataset_cfg, task_cfg, model_cfg, trainer_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():

    # Set seed
    seed_everything(2299)

    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup(None)

    # Create network
    model_cfg["seq_len"] = dm.train_dataset.n_genes + 1
    model_cfg["class_dist"] = dm.train_dataset.class_dist

    model = Exceiver(**model_cfg)
    model.to(device=device)

    task = MSELoss(
        network=model,
        task_cfg=task_cfg,
    )


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=1000,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        # callbacks=[early_stop, checkpoint_callback],
    )

    trainer.fit(task, dm)



if __name__ == "__main__":

    main()