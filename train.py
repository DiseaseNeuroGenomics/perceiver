
import torch
import pickle
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, GatedMLP
from tasks import MSELoss, AdverserialLoss
from config_immune import dataset_cfg, task_cfg, model_cfg, trainer_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')
#torch.backends.cuda.sdp_kernel(True)
#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_math_sdp(True)

def check_train_test_set(cfg):
    train = pickle.load(open(cfg["train_metadata_path"], "rb"))
    test = pickle.load(open(cfg["test_metadata_path"], "rb"))
    train_ids = set(train["obs"]["SubID"])
    test_ids = set(test["obs"]["SubID"])
    print(f"Number of subjects in train set: {len(train_ids)}, and test set: {len(test_ids)}")
    train_test_inter = train_ids.intersection(test_ids)
    print(f"Number of train users in test set: {len(train_test_inter)}")


def main(train_idx, test_idx):

    # Set seed
    pl.seed_everything(42)

    # check_train_test_set(dataset_cfg)

    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx

    #dataset_cfg["cell_restrictions"] = {"class": cell_class, }

    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")
    task_cfg["gene_names"] = dm.train_dataset.gene_names

    # Set up data module
    dm = DataModule(**dataset_cfg)

    # Transfer information from Dataset
    model_cfg["seq_len"] = dm.n_genes
    model_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_bins"] = dataset_cfg["n_bins"]
    task_cfg["cell_properties"] = dm.cell_properties

    # Create network
    model = Exceiver(**model_cfg)
    # model = GatedMLP(**model_cfg)

    for n, p in model.named_parameters():
        print(n, p.size())

    task = MSELoss(network=model, task_cfg=task_cfg)

    trainer = pl.Trainer(
        enable_checkpointing=True,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=1000,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=10_000,
        limit_val_batches=2_000,
        check_val_every_n_epoch=1,
    )

    trainer.fit(task, dm)


if __name__ == "__main__":

    splits = pickle.load(open("/home/masse/work/data/data_for_perturb/splits_full.pkl", "rb"))

    #for split_num in range(0, 20):

    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]
    main(train_idx, test_idx)