from typing import Any, Dict, List, Optional

import os, shutil
import copy
import pickle
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ExplainedVariance, MeanSquaredError
from torchmetrics.classification import Accuracy, BinaryAccuracy
import torch.nn.functional as F
from torch.distributions.normal import Normal
from losses import FocalLoss



class BaseTask(pl.LightningModule):

    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):

        # Initialize superclass
        super().__init__()
        self.network = network
        self.task_cfg = task_cfg
        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self.source_code_copied = False

    def _copy_source_code(self):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        base_dir = os.getcwd()
        print(f"Base dir: {base_dir}")
        src_files = [
            f"{base_dir}/config.py",
            f"{base_dir}/config_tf.py",
            f"{base_dir}/datasets.py",
            f"{base_dir}/networks.py",
            f"{base_dir}/tasks.py",
            f"{base_dir}/train.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")

    def training_step(self, batch, batch_idx):

        return None

    def validation_step(self, batch, batch_idx):

        return None

    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1

    def configure_optimizers(self):

        return torch.optim.AdamW(
            self.network.parameters(),
            lr=self.task_cfg["learning_rate"],
            weight_decay=self.task_cfg["weight_decay"],
        )

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        closure,
        on_tpu=None,
        using_native_amp=None,
        using_lbfgs=None,
        min_lr=1e-8,
    ):

        # warm up lr
        if self.trainer.global_step < self.task_cfg["warmup_steps"]:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.task_cfg["warmup_steps"]))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]


        elif self.trainer.global_step > self.task_cfg["warmup_steps"]:
            lr_scale = self.task_cfg["decay"] ** (self.trainer.global_step - self.task_cfg["warmup_steps"])
            lr_scale = max(min_lr, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]

        # update params
        optimizer.step(closure=closure)


class RecursiveInference(BaseTask):
    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):

        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            if not "feature" in n:
                p.requires_grad_(False)

        self._load_gene_perturb_info()
        self._create_results_dict()
        self.source_code_copied = False

    def _load_gene_perturb_info(self):
        self.gene_perturb = self.task_cfg["gene_perturb"]
        self.gene_perturb_idx = np.where(np.array(self.gene_names) == self.gene_perturb)[0][0]

    def _create_results_dict(self):

        n_genes = len(self.gene_names)
        n_cells = len(self.task_cfg["cell_idx"])
        """
        self.results = {
            "gene_names": self.gene_names,
            "gene_perturb": self.gene_perturb,
            "gene_perturb_idx": self.gene_perturb_idx,
            "cell_idx": - np.ones((n_cells, n_genes), dtype=np.int16),
            "pred_exp": np.zeros((n_cells, n_genes), dtype=np.float32),
            "actual_exp": np.zeros((n_cells, n_genes), dtype=np.float32),
        }
        """


    def _subsample_batch(self, batch):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths, _, _, cell_idx = batch

        # search for out perturb in the input
        perturb_idx = torch.where(gene_ids == self.gene_perturb_idx)
        if len(perturb_idx[0]) < 4:
            # too few samples, don't both
            return None, None


        batch = (
            gene_ids[perturb_idx[0]],
            gene_target_ids[perturb_idx[0]],
            gene_vals[perturb_idx[0]],
            gene_targets[perturb_idx[0]],
            key_padding_mask[perturb_idx[0]],
            depths[perturb_idx[0]] if depths is not None else None,
            [cell_idx[i] for i in perturb_idx[0]],
        )

        perturb_idx = torch.where(batch[0] == self.gene_perturb_idx)
        return batch, perturb_idx

    def _get_possible_gene_index(self, batch):

        gene_ids = batch[0].detach().cpu().numpy()
        gene_target_ids = batch[1].detach().cpu().numpy()
        idx = list(gene_ids.flatten()) + list(gene_target_ids.flatten())
        idx = list(set(idx) - set([self.gene_perturb_idx]))
        return np.unique(idx)


    def validation_step(self, batch, batch_idx):

        n_steps = 20
        n_genes = len(self.gene_names)
        possible_gene_idx = self._get_possible_gene_index(batch)

        batch, perturb_idx = self._subsample_batch(batch)
        if batch is None: # happens where there are too few samples containing the gene to perturb
            return

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths, cell_idx = batch
        batch_size, n_input_genes = gene_ids.shape
        perturb_increase = 1.0


        perturb_gene_clamp_val = gene_vals[perturb_idx] + perturb_increase

        for i in range(len(cell_idx)):
            pass
            # self.results["cell_idx"].append(cell_idx[i])
            #self.results["actual_exp"][cell_idx[i], gene_ids[i].detach().cpu().numpy()] = gene_vals[i].detach().cpu().numpy()
            #self.results["actual_exp"][cell_idx[i], gene_target_ids[i].detach().cpu().numpy()] = gene_targets[i].detach().cpu().numpy()

        new_gene_vals = copy.deepcopy(gene_vals)
        current_gene_vals = torch.zeros((batch_size, n_genes), dtype=torch.float32).to(gene_ids.device)
        new_gene_vals[perturb_idx] = perturb_gene_clamp_val

        #gene_target_ids = torch.from_numpy(possible_gene_idx)[None, :].repeat(batch_size, 1).to(gene_ids.device)
        gene_target_ids = gene_ids

        for iter in range(n_steps):

            print("before", iter, new_gene_vals[0, :].mean())
            gene_pred, _, _ = self.network.forward(
                gene_ids, gene_target_ids, new_gene_vals, key_padding_mask, depths,
            )

            alpha = torch.clip(gene_pred[:, :, 0].to(torch.float32), 0, 100)
            gene_pred = torch.clip(gene_pred[:, :, 0].to(torch.float32), 0, 100)
            #gene_pred = torch.log(1 + torch.poisson(torch.exp(gene_pred) - 1))


            #gene_pred = torch.log(1 + torch.poisson(torch.exp(gene_pred) - 1))
            print("after", iter, new_gene_vals[0, :].mean(), gene_pred[0, :].mean(), alpha[0, :].mean())


            for n in range(batch_size):
                # order matters
                current_gene_vals[n, gene_ids[n]] = copy.deepcopy(new_gene_vals[n, :])
                current_gene_vals[n, gene_target_ids[n]] = copy.deepcopy(gene_pred[n, :].to(torch.float32))

            y0 = gene_pred.to(torch.float32).detach().cpu().numpy()
            y1 = gene_targets.detach().cpu().numpy()
            y2 = current_gene_vals.detach().cpu().numpy()
            y3 = new_gene_vals.detach().cpu().numpy()

            """
            print("QQQ", type(gene_ids))
            print("QQQ", gene_ids[0])

            idx = gene_ids[0].detach().cpu().numpy()
            print("DD", idx.shape)
            print("A", np.mean(y0[0, :]))
            print("B", np.mean(y1[0, :]))
            print("C", np.mean(y2[0, idx]))
            print("D", np.mean(y3[0, :]))

            print("A", current_gene_vals[0, gene_ids[0]].mean())
            print("B", new_gene_vals[0, :].mean())
            print("C", gene_pred[0, :].mean())
            print("D", gene_targets[0, :].mean())
            """


            for i in range(batch_size):
                gene_ids[i, 0] = self.gene_perturb_idx
                new_gene_vals[i, 0] = perturb_gene_clamp_val[i]

                #idx_input = np.random.choice(possible_gene_idx, n_input_genes - 1, replace=False)
                idx_input = gene_ids[i, :]
                #gene_ids[i, 1:] = torch.from_numpy(idx_input).to(gene_ids.device)
                #new_gene_vals[i, 1:] = copy.deepcopy(current_gene_vals[i, idx_input])
                new_gene_vals[i, :] = copy.deepcopy(current_gene_vals[i, idx_input])
                #remaining_idx = list(set(possible_gene_idx) - set(idx_input) - set([self.gene_perturb_idx]))
                #replace = False if len(remaining_idx) >= n_target_genes else True
                #idx_target = np.random.choice(possible_gene_idx, n_target_genes, replace=replace)
                #gene_target_ids[i, :] = torch.from_numpy(idx_target).to(gene_ids.device)

        current_gene_vals = current_gene_vals.detach().cpu().numpy()
        for i in range(len(cell_idx)):
            self.results["pred_exp"][cell_idx[i], :] = current_gene_vals[i, :]
            # self.results["pred_exp"][cell_idx[i], gene_target_ids[i].detach().cpu().numpy()] = current_gene_vals[i].detach().cpu().numpy()




class TFInference(BaseTask):
    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):

        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            if not "feature" in n:
                p.requires_grad_(False)

        self._load_tf_info()
        self._create_results_dict()
        self.source_code_copied = False

    def _load_tf_info(self):
        self.tf_list = pd.read_csv(self.task_cfg["tf_list_fn"], header=None)[0].values
        self.tf_list_idx = []
        for tf in self.tf_list:
            if tf in self.gene_names:
                self.tf_list_idx.append(np.where(np.array(self.gene_names) == tf)[0][0])


    def _create_results_dict(self):

        n_tfs = len(self.tf_list)
        n_cells = len(self.task_cfg["cell_idx"])
        self.results = {
            "gene_names": self.gene_names,
            "tf_list": self.tf_list,
            "tf_list_idx": self.tf_list_idx,
            "cell_idx": self.task_cfg["cell_idx"],
            "counts": np.zeros((n_cells, n_tfs), dtype=np.int16),
            "pred_tf_exp": np.zeros((n_cells, n_tfs), dtype=np.float32),
            "actual_tf_exp": np.zeros((n_cells, n_tfs), dtype=np.float32),
        }


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths, _, _,  cell_idx = batch
        batch_size, n_target_genes = gene_target_ids.shape

        gene_pred, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        gene_target_ids = gene_target_ids.detach().cpu().numpy()
        gene_target_ids = np.reshape(gene_target_ids, (-1))
        gene_pred = gene_pred.to(torch.float32).detach().cpu().numpy()
        gene_pred = np.reshape(gene_pred, (-1))

        gene_targets = gene_targets.to(torch.float32).detach().cpu().numpy()
        gene_targets = np.reshape(gene_targets, (-1))

        cell_idx = np.array(cell_idx)
        cell_idx = np.tile(cell_idx[:, None], (1, n_target_genes))
        cell_idx = np.reshape(cell_idx, (-1))

        for j, i in enumerate(self.tf_list_idx):
            if i in gene_target_ids:
                idx = np.where(gene_target_ids == i)[0]
                rows = cell_idx[idx]
                #cols = gene_target_ids[idx]
                self.results["counts"][rows, j] += 1
                self.results["pred_tf_exp"][rows, j] += gene_pred[idx]
                self.results["actual_tf_exp"][rows, j] = gene_targets[idx]


class AllToAllPerutbation(BaseTask):

    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg)

        for n, p in self.network.named_parameters():
            p.requires_grad_(False)

        self._create_results_dict()


    def _create_results_dict(self):

        self.results = {"gene_names": self.gene_names}
        N = len(self.gene_names)
        self.results["pos_delta"] = np.zeros((N, N), dtype=np.float32)
        self.results["pos_var"] = np.zeros((N, N), dtype=np.float32)
        self.results["pos_count"] = np.zeros((N, N), dtype=np.uint16)

        #self.results["neg_delta"] = np.zeros((N, N), dtype=np.float32)
        #self.results["neg_var"] = np.zeros((N, N), dtype=np.float32)
        #self.results["neg_count"] = np.zeros((N, N), dtype=np.uint16)


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths, _, _, _ = batch
        batch_size = gene_ids.shape[0]
        n_iters = 10

        n_input_genes = gene_vals.shape[1]

        gene_pred, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        target_idx = gene_target_ids.to(torch.int16).detach().cpu().numpy()
        rnd_idx = np.random.choice(n_input_genes, size=(10,), replace=False)

        for n in range(n_iters):

            # rnd_inc = torch.from_numpy(np.random.choice([1, 2, 3], size=(batch_size,), replace=True)).to(gene_vals.device)

            gene_vals_perturb = copy.deepcopy(gene_vals.clone().detach())
            gene_vals_perturb[:, rnd_idx[n]] = gene_vals_perturb[:, rnd_idx[n]] + 1
            # TODO: add fix when binning inputs
            #gene_vals_perturb[:, rnd_idx[n]] = torch.clip(
            #    gene_vals_perturb[:, rnd_idx[n]] + rnd_inc, 0, self.n_bins-1
            #)

            gene_pred_perturb, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, key_padding_mask, depths,
            )


            source_idx = gene_ids[:, rnd_idx[n]].to(torch.int16).detach().cpu().numpy()
            delta = gene_pred_perturb.to(torch.float32).detach().cpu().numpy() - gene_pred.to(torch.float32).detach().cpu().numpy()
            delta = delta[:, :, 0]

            for i in range(batch_size):
                count = self.results["pos_count"][source_idx[i], target_idx[i]]
                prev_mean = self.results["pos_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)
                prev_var = self.results["pos_var"][source_idx[i], target_idx[i]]

                self.results["pos_count"][source_idx[i], target_idx[i]] += 1
                self.results["pos_delta"][source_idx[i], target_idx[i]] += delta[i, :]

                count = self.results["pos_count"][source_idx[i], target_idx[i]]
                current_mean = self.results["pos_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)

                current_var = prev_var + (delta[i, :] - prev_mean) * (delta[i, :] - current_mean)
                self.results["pos_var"][source_idx[i], target_idx[i]] = current_var
        """
        for n in range(n_iters):

            gene_vals_perturb = copy.deepcopy(gene_vals.clone().detach())
            rnd_idx = np.zeros(batch_size, dtype=np.int64)
            for i in range(batch_size):
                idx = torch.where(gene_vals_perturb[i, :] >= 1)[0]
                if len(idx) > 0:
                    rnd_idx[i] = np.random.choice(idx.cpu().numpy())
                    #gene_vals_perturb[i, rnd_idx[i]] = torch.clip(
                    #    gene_vals_perturb[i, rnd_idx[i]] - 1, 0, self.n_bins-1
                    #)
                    gene_vals_perturb[i, rnd_idx[i]] = 0


            gene_pred_perturb, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, key_padding_mask,
            )

            source_idx = gene_ids[torch.arange(batch_size), rnd_idx].to(torch.int16).detach().cpu().numpy()


            delta = gene_pred_perturb.to(torch.float32).detach().cpu().numpy() - gene_pred.to(torch.float32).detach().cpu().numpy()
            delta = delta[:, :, 0]
            #print("B", delta.std(), torch.sum(torch.abs(gene_vals_perturb - gene_vals)))

            for i in range(batch_size):
                count = self.results["neg_count"][source_idx[i], target_idx[i]]
                prev_mean = self.results["neg_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)
                prev_var = self.results["neg_var"][source_idx[i], target_idx[i]]

                self.results["neg_count"][source_idx[i], target_idx[i]] += 1
                self.results["neg_delta"][source_idx[i], target_idx[i]] += delta[i, :]

                count = self.results["neg_count"][source_idx[i], target_idx[i]]
                current_mean = self.results["neg_delta"][source_idx[i], target_idx[i]] / (1e-6 + count)

                current_var = prev_var + (delta[i, :] - prev_mean) * (delta[i, :] - current_mean)
                self.results["neg_var"][source_idx[i], target_idx[i]] = current_var
        """


class MSELoss(BaseTask):
    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg)

        # Functions and metrics
        self.mse = nn.MSELoss()
        self.val_mse = MeanSquaredError()
        self.explained_var = ExplainedVariance()

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self._create_results_dict()
        self.source_code_copied = False


    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        self.results_list = ["gene_target_ids", "gene_targets", "gene_pred"]
        for k in self.results_list:
            self.results[k] = []


    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths,_, _, _ = batch
        gene_pred, latent, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))

        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return gene_loss


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask, depths, _, _, _ = batch

        gene_pred, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        if self.task_cfg["save_predictions"]:
            self.results["gene_target_ids"].append(gene_target_ids.to(torch.int16).detach().cpu().numpy())
            self.results["gene_targets"].append(gene_targets.detach().cpu().numpy())
            self.results["gene_pred"].append(gene_pred.to(torch.float32).detach().cpu().numpy())

        self.explained_var(gene_pred, gene_targets.unsqueeze(2))
        self.val_mse(gene_pred, gene_targets.unsqueeze(2))

        self.log("exp_var", self.explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1
        for k in self.results_list:
            self.results[k] = []


class Annotation(pl.LightningModule):
    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):

        # Initialize superclass
        super().__init__()
        self.network = network

        for n, p in self.network.named_parameters():

            #if not ("feature" in n or "decoder" in n or "31" in n or "30" in n or "29" in n or "28" in n): # and not cond:
            if not "feature" in n:
                p.requires_grad_(False)

        for k, v in task_cfg.items():
            setattr(self, k, v)

        self.task_cfg = task_cfg
        self.cell_properties = task_cfg["cell_properties"]

        # Functions and metrics
        self._cell_properties_metrics()

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None

        self.n_bins = task_cfg["n_bins"] if "n_bins" in task_cfg else None
        self.scale_target_depth = 3.0


        self._create_results_dict()
        self.source_code_copied = False

    def _cell_properties_metrics(self):

        self.cell_cross_ent = nn.ModuleDict()
        self.cell_mse = nn.ModuleDict()
        self.mse = nn.MSELoss()
        self.cell_accuracy = nn.ModuleDict()
        self.cell_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                #weight = torch.from_numpy(
                #    np.float32(np.clip(1 / cell_prop["freq"], 0.0001, 10000.0))
                #) if self.balance_classes else None

                weight = torch.from_numpy(
                    np.float32(np.clip((np.max(cell_prop["freq"]) / cell_prop["freq"]), 1.0, 25.0))
                ) if self.balance_classes else None


                #print("Weight", weight)
                self.cell_cross_ent[k] = nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=-100)
                #self.cell_cross_ent[k] = FocalLoss(len(cell_prop["values"]), gamma=2.0, alpha=2.0)
                self.cell_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_mse[k] = nn.MSELoss(reduction="none")
                self.cell_explained_var[k] = ExplainedVariance()

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                n = len(cell_prop["values"])
                self.results[f"{k}_matrix"] = np.zeros((n, n))



    def _copy_source_code(self):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        base_dir = "/home/masse/work/perceiver/"
        src_files = [
            f"{base_dir}config.py",
            f"{base_dir}datasets.py",
            f"{base_dir}networks.py",
            f"{base_dir}tasks.py",
            f"{base_dir}train.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")

    def _feature_scores(self, cell_pred, cell_targets, cell_mask):

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            idx = torch.nonzero(cell_mask[:, n])

            if cell_prop["discrete"]:
                pred_idx = torch.argmax(cell_pred[k], dim=-1).to(torch.int64)
                # pred_prob = F.softmax(cell_pred[k], dim=-1).to(torch.float32).detach().cpu().numpy()
                targets = cell_targets[:, n].to(torch.int64)
                self.cell_accuracy[k].update(pred_idx[idx, 0], targets[idx])

                idx1 = pred_idx[idx, 0].cpu().numpy()
                idx0 = targets[idx].cpu().numpy()
                for i, j in zip(idx0, idx1):
                    self.results[f"{k}_matrix"][i, j] += 1

            else:
                pred = cell_pred[k][:, 0]
                try: # rare error
                    self.cell_explained_var[k].update(pred[idx], cell_targets[idx, n])
                except:
                    self.cell_explained_var[k].update(pred, cell_targets[idx, n])


        for k, v in self.cell_accuracy.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_explained_var.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def _feature_loss(
        self,
        cell_pred: Dict[str, torch.Tensor],
        cell_prop_vals: torch.Tensor,
        cell_mask: torch.Tensor,
    ):

        cell_loss = 0.0
        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                loss = self.cell_cross_ent[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n].to(torch.int64))
                cell_loss += (loss * cell_mask[:, n]).mean()
            else:
                loss = self.cell_mse[k](torch.squeeze(cell_pred[k]), cell_prop_vals[:, n])
                cell_loss += (loss * cell_mask[:, n]).mean()

        if self.training:
            self.log("cell_loss_train", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("cell_loss_val", cell_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return cell_loss

    def training_step(self, batch, batch_idx):

        (
            gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask,
            depths, cell_prop_vals, cell_prop_mask, _
        ) = batch

        # depths[:, 0] = depths[:, 0] * self.scale_target_depth
        d = torch.exp(depths[:, 0])
        depths[:, 0] = torch.log(d * 3)

        gene_pred, _, feature_pred = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths,
        )

        #print("TRAIN", gene_pred.mean())

        feature_loss = self._feature_loss(feature_pred, cell_prop_vals, cell_prop_mask)
        #gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        gene_loss = 0.0

        #self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("feature_loss", feature_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return feature_loss


    def validation_step(self, batch, batch_idx):

        (
            gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask,
            depths, cell_prop_vals, cell_prop_mask, _
        ) = batch

        d = torch.exp(depths[:, 0])
        depths[:, 0] = torch.log(d * 3)

        gene_pred, _, feature_pred = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask, depths
        )
        #print("VAL", gene_pred.mean())

        self._feature_scores(feature_pred, cell_prop_vals, cell_prop_mask)
        self._feature_loss(feature_pred, cell_prop_vals, cell_prop_mask)


    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"

        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1
        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                n = len(cell_prop["values"])
                self.results[f"{k}_matrix"] = np.zeros((n, n))

    def configure_optimizers(self):

        #params = [p for n, p in self.network.named_parameters() if not "gmlp" in n]
        out_params = [p for n, p in self.network.named_parameters() if "feature_decoder.cell_type.gene_mlp" in n]
        other_params = [p for n, p in self.network.named_parameters() if "feature_decoder.cell_type.decoder_cross_attn" in n]


        return torch.optim.AdamW(
            #[
            #    {"params": out_params0, "lr": self.task_cfg["learning_rate"]},
            #    {"params": params1, "lr": self.task_cfg["learning_rate"]},
            #],
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
        )




class MSELoss_atac(pl.LightningModule):
    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):

        # Initialize superclass
        super().__init__()
        self.network = network
        self.task_cfg = task_cfg


        self.cell_prop_cross_ent = None
        self.cell_prop_accuracy = None
        self.cell_prop_mse = None
        self.cell_prop_explained_var = None

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self.n_bins = task_cfg["n_bins"] if "n_bins" in task_cfg else None

        if self.task_cfg["predict_atac"]:
            self.cross_ent = nn.BCEWithLogitsLoss(weight=None)
            self.atac_accuracy = Accuracy(task='binary', average='macro', num_classes=2)
        else:
            self.mse = nn.MSELoss()
            self.gene_explained_var = ExplainedVariance()
            self.metrics = MetricCollection([ExplainedVariance()])


        self._create_results_dict()
        self.source_code_copied = False

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        # N = len(self.gene_names)
        N = 16594
        #self.results["target_gene_counts"] = np.zeros(N)
        #self.results["target_gene_preds"] = np.zeros(N)
        #self.results["atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        #self.results["pred_atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        #self.gcount = np.zeros(N)

    def _copy_source_code(self):

        target_dir = f"{self.trainer.log_dir}/code"
        os.mkdir(target_dir)
        base_dir = "/home/masse/work/perceiver/"
        src_files = [
            f"{base_dir}config_atac.py",
            f"{base_dir}datasets.py",
            f"{base_dir}networks.py",
            f"{base_dir}tasks.py",
            f"{base_dir}train_atac.py",
        ]
        for src in src_files:
            shutil.copyfile(src, f"{target_dir}/{os.path.basename(src)}")

    def training_step(self, batch, batch_idx):

        # atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        # gene_pred, latent = self.network.forward(
        #    atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, padding_mask_atac,
        #    padding_mask_genes,
        # )
        atac_gene_based, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        gene_pred, atac_pred, latent = self.network.forward(
            atac_gene_based, gene_ids, gene_vals, gene_target_ids, padding_mask_atac, padding_mask_genes,
        )

        if self.task_cfg["predict_atac"]:
            loss = self.cross_ent(atac_pred, atac_gene_based.squeeze())
        else:
            loss = self.mse(gene_pred, gene_target_vals.unsqueeze(2))

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):

        #atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        #gene_pred, latent = self.network.forward(
        #    atac_chr, atac_pos, atac_pos_abs, gene_ids, gene_vals, gene_target_ids, padding_mask_atac,
        #    padding_mask_genes,
        #)
        atac_gene_based, gene_ids, gene_vals, gene_target_ids, gene_target_vals, padding_mask_atac, padding_mask_genes = batch
        gene_pred, atac_pred, latent = self.network.forward(
            atac_gene_based, gene_ids, gene_vals, gene_target_ids, padding_mask_atac,padding_mask_genes,
        )
        """
        for i in range(gene_pred.shape[0]):
            idx = gene_target_ids[i, :].detach().cpu().numpy()
            y = np.squeeze(gene_pred[i, :].to(torch.float32).detach().cpu().numpy())
            self.results["target_gene_counts"][idx] += 1
            self.results["target_gene_preds"][idx] += y
        """

        #self.results["gene_target_ids"].append(gene_target_ids.to(torch.int16).detach().cpu().numpy())
        #self.results["gene_vals"].append(gene_vals.detach().cpu().numpy())
        #self.results["gene_ids"].append(gene_ids.detach().cpu().numpy())
        #self.results["gene_targets"].append(gene_targets.detach().cpu().numpy())
        #self.results["gene_pred"].append(gene_pred.to(torch.float32).detach().cpu().numpy())
        #self.results["atac_data"].append(atac_gene_based.to(torch.float32).detach().cpu().numpy())
        #self.results["pred_atac_data"].append(atac_pred.to(torch.float32).detach().cpu().numpy())

        """
        x0 = atac_gene_based.to(torch.float32).detach().cpu().numpy()
        y0 = atac_pred.to(torch.float32).detach().cpu().numpy()

        for i in range(48):
            idx = gene_target_ids[i, :].detach().cpu().numpy()
            for k, j in enumerate(idx):
                if self.gcount[j] < 199:

                    #print(self.count[j], j)
                    self.results["atac_data"][int(self.gcount[j]), j, :] += x0[i, k, 0, :]
                    self.results["pred_atac_data"][int(self.gcount[j]), j, :] += y0[i, k]
                    self.gcount[j] += 1

        """

        #self.results["atac_data"] += atac_gene_based[gene_target_ids.to(torch.float32).detach().cpu().numpy()

        if self.task_cfg["predict_atac"]:
            # print("AAAA", atac_pred.shape, atac_gene_based.shape)
            self.atac_accuracy(atac_pred, atac_gene_based.squeeze())
            self.log("atac_acc", self.atac_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.gene_explained_var(gene_pred, gene_target_vals.unsqueeze(2))
            self.log("gene_ex", self.gene_explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        """
        if self.perturb_idx is None:
            fn = f"{self.trainer.log_dir}/test_results.pkl"
        else:
            fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    if len(self.results[k]) > 0:
        #        self.results[k] = np.concatenate(self.results[k], axis=0)
        
        self.results["epoch"] = self.current_epoch + 1
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        N = len(self.gene_names)
        #self.results["target_gene_counts"] = np.zeros(N)
        #self.results["target_gene_preds"] = np.zeros(N)
        """
        #self.results["atac_data"] = np.mean(np.stack(self.results["atac_data"]), axis=(0,1))
        #self.results["pred_atac_data"] = np.mean(np.stack(self.results["pred_atac_data"]), axis=(0, 1))
        fn = f"{self.trainer.log_dir}/test_results.pkl"
        #pickle.dump(self.results, open(fn, "wb"))
        N = 16594
        # self.results["target_gene_counts"] = np.zeros(N)
        # self.results["target_gene_preds"] = np.zeros(N)
        #self.results["atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        #self.results["pred_atac_data"] = np.zeros((200, N, 128), dtype=np.float32)
        #self.gcount = np.zeros(N)



    def configure_optimizers(self):

        base_params = [p for n, p in self.network.named_parameters() if "SubID" not in n]
        reversal_params = [p for n, p in self.network.named_parameters() if "SubID" in n]

        return torch.optim.AdamW(
            [
                {"params": base_params},
                {"params": reversal_params, "lr": self.task_cfg["learning_rate"] * 10, "weight_decay": 0.0},
            ],
            lr=self.task_cfg["learning_rate"],
            weight_decay=self.task_cfg["weight_decay"],
        )

    def optimizer_step(
            self,
            current_epoch,
            batch_nb,
            optimizer,
            closure,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
            min_lr=1e-8,
    ):

        # warm up lr
        if self.trainer.global_step < self.task_cfg["warmup_steps"]:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1) / float(self.task_cfg["warmup_steps"]),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]


        elif self.trainer.global_step > self.task_cfg["decay_steps"]:
            lr_scale = self.task_cfg["decay"] ** (self.trainer.global_step - self.task_cfg["decay_steps"])
            lr_scale = max(min_lr, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_cfg["learning_rate"]


        # update params
        optimizer.step(closure=closure)


class AdverserialLoss(MSELoss):
    def __init__(
        self,
        network,
        task_cfg,
        adv_cell_prop: str = "SubID",
        adv_loss_ratio: float = 0.05,
        adv_threshold: float = 0.0,
        **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg, automatic_optimization=False, **kwargs)
        print("ADV LOSS COEFF", adv_loss_ratio)
        self.network_params = [p for n, p in self.network.named_parameters() if adv_cell_prop not in n]
        self.adv_params = [p for n, p in self.network.named_parameters() if adv_cell_prop in n]
        self.adv_cell_prop = adv_cell_prop
        self.adv_loss_ratio = adv_loss_ratio
        self.adv_threshold = adv_threshold

        self.adv_accuracy = nn.ModuleDict()
        self.adv_accuracy[adv_cell_prop] = Accuracy(
            task="multiclass", num_classes=len(self.cell_properties[adv_cell_prop]["values"]), average="macro",
        )
        self.steps = 0.0

    def training_step(self, batch, batch_idx):

        opt0, opt1 = self.configure_optimizers()
        opt0, opt1 = self.update_lr(opt0, opt1)
        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets, cell_class_id = batch
        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, cell_class_id, key_padding_mask,
        )

        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        cell_prop_loss = 0.0
        adv_loss = 0.0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):

            if cell_prop["discrete"]:
                # discrete variable, use cross entropy
                # class values of -1 will be masked out
                idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                p_loss = self.cell_prop_cross_ent[k](
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n].to(torch.int64)
                )

                if k == self.adv_cell_prop:
                    if p_loss < self.adv_threshold:
                        cell_prop_loss -= self.adv_loss_ratio * p_loss
                    opt1.zero_grad()
                    self.manual_backward(p_loss, retain_graph=True)
                    self.log(
                        "adv_acc", p_loss,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
                else:
                    cell_prop_loss += p_loss

            else:
                # continuous variable, use MSE
                # class values less than -999 or greater than 999 will be masked out
                idx = torch.where(cell_prop_targets[:, n] > -999)[0]
                cell_prop_loss += self.cell_prop_mse[k](
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n]
                )

        loss = self.task_cfg["gene_weight"] * gene_loss + self.task_cfg["cell_prop_weight"] * cell_prop_loss

        opt0.zero_grad()
        self.manual_backward(loss)

        # clip gradients
        self.clip_gradients(opt0, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(opt1, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # apply gradient step
        opt0.step()
        opt1.step()

        # log results
        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "cell_loss",
            cell_prop_loss + self.adv_loss_ratio * adv_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "adv_acc", adv_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


    def on_train_epoch_end(self):
        """Need to manually save checkpoints as automatic checkpointing is disabled for some reason..."""
        ckpt_path = self.logger.log_dir + "/saved_model.ckpt"
        self.trainer.save_checkpoint(ckpt_path)


    def update_lr(self, opt0, opt1):

        self.steps += 1.0
        lr = self.task_cfg["learning_rate"] * np.minimum(
            self.steps ** (-0.5), self.steps * float(self.task_cfg["warmup_steps"]) ** (-1.5))
        for opt in [opt0, opt1]:
            for pg in opt.param_groups:
                pg["lr"] = lr
        return opt0, opt1


    def configure_optimizers(self):

        opt0 = torch.optim.AdamW(
            self.network_params,
            lr=0.0,
            weight_decay=self.task_cfg["weight_decay"],
        )
        opt1 = torch.optim.AdamW(
            self.adv_params,
            lr=0.0,
            weight_decay=self.task_cfg["weight_decay"],
         )

        return opt0, opt1
