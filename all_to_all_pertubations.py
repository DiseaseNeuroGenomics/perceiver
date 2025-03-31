
import torch
import pickle
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, GatedMLP, load_model
from tasks import AllToAllPerutbation
from config_immune import dataset_cfg, task_cfg, model_cfg, trainer_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')

def check_train_test_set(cfg):
    train = pickle.load(open(cfg["train_metadata_path"], "rb"))
    test = pickle.load(open(cfg["test_metadata_path"], "rb"))
    train_ids = set(train["obs"]["SubID"])
    test_ids = set(test["obs"]["SubID"])
    print(f"Number of subjects in train set: {len(train_ids)}, and test set: {len(test_ids)}")
    train_test_inter = train_ids.intersection(test_ids)
    print(f"Number of train users in test set: {len(train_test_inter)}")


def main(train_idx, test_idx, model_save_path, cell_class=None):

    # Set seed
    pl.seed_everything(45)

    # check_train_test_set(dataset_cfg)
    #dataset_cfg["cell_restrictions"] = {"class": cell_class}


    #dataset_cfg["cell_restrictions"] = {"perturbation": "control", "experiment": "Norman_2019"}
    dataset_cfg["cell_restrictions"] = None
    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx
    dataset_cfg["test_idx"] = test_idx
    dataset_cfg["perturbation"] = None

    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")
    task_cfg["gene_names"] = dm.train_dataset.gene_names
    task_cfg["n_bins"] = dataset_cfg["n_bins"]


    # Set up data module
    dm = DataModule(**dataset_cfg)

    # Transfer information from Dataset
    model_cfg["seq_len"] = dm.n_genes
    model_cfg["cell_properties"] = dm.cell_properties
    task_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_bins"] = dataset_cfg["n_bins"]
    task_cfg["learning_rate"] = 0.0

    # Create network
    model = Exceiver(**model_cfg)
    #model = GatedMLP(**model_cfg)

    model = load_model(model_save_path, model)

    for n, p in model.named_parameters():
        print(n, p.size())

    task = AllToAllPerutbation(network=model, task_cfg=task_cfg)

    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=1000,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        limit_train_batches=1,
        limit_val_batches=200,
    )

    trainer.fit(task, dm)


if __name__ == "__main__":

    #splits = pickle.load(open("/home/masse/work/data/psychAD_PD/train_test_20splits.pkl", "rb"))
    #model_save_path = "/home/masse/work/data/perceiver_perturbations/version_18/checkpoints/epoch=23-step=120000.ckpt"
    cell_class = None

    model_save_path = "/home/masse/work/perceiver/lightning_logs/version_120/checkpoints/epoch=24-step=125000.ckpt"
    splits = pickle.load(open("/home/masse/worchek/data/data_for_perturb/splits_full.pkl", "rb"))

    for split_num in range(0, 1):

        train_idx = splits["norman_virus_pert_test_idx"]
        test_idx = splits["norman_virus_pert_test_idx"]

        main(train_idx, test_idx, model_save_path, cell_class)
