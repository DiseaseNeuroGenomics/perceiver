import os, shutil
import copy
import pickle
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ExplainedVariance, MeanSquaredError
from torchmetrics.classification import Accuracy, BinaryAccuracy
import torch.nn.functional as F
from torch.distributions.normal import Normal


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):

        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        loss = - target * log_prob * (1 - prob) ** self.gamma
        return loss.sum(dim=-1).mean()

class ContrastiveLoss(pl.LightningModule):

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
        self.cell_properties = self.task_cfg["cell_properties"]

        self.cell_prop_cross_ent = nn.ModuleDict()
        self.cell_prop_mse = nn.ModuleDict()
        self.cell_prop_accuracy = nn.ModuleDict()
        self.cell_prop_explained_var = nn.ModuleDict()

        for k, cell_prop in self.cell_properties.items():
            if cell_prop["discrete"]:
                # discrete variable, set up cross entropy module
                weight = torch.from_numpy(
                    np.float32(np.clip(1 / cell_prop["freq"], 0.1, 10.0))
                ) if task_cfg["balance_classes"] else None
                self.cell_prop_cross_ent[k] = nn.CrossEntropyLoss(weight=weight)
                # self.cell_prop_cross_ent[k] = FocalLoss(gamma = 2.0)
                self.cell_prop_accuracy[k] = Accuracy(
                    task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                )
            else:
                # continuous variable, set up MSE module
                self.cell_prop_mse[k] = nn.MSELoss()
                self.cell_prop_explained_var[k] = ExplainedVariance()

        #self.criterion = nn.CrossEntropyLoss()

        self._create_results_dict()

        self.temperature = 0.5

    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []
            self.results["class_id"] = []

    def info_nce_loss(self, z0, z1, labels):

        # zo and z1 have shape [Batch, seq_dim]
        # normalize
        z0 = z0 / torch.norm(z0, p=2, dim=-1, keepdim=True)
        z1 = z1 / torch.norm(z1, p=2, dim=-1, keepdim=True)

        batch_size = z1.size(0)
        n_views = 2
        n_labels = batch_size * n_views

        features = torch.cat([z0, z1], dim=0)

        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z0.device)
        """
        labels = torch.cat([labels, labels], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        similarity_matrix = torch.matmul(features, features.T)

        #print("AA", labels.size(), similarity_matrix.size())
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(n_labels, dtype=torch.bool).to(z0.device)
        labels = labels[~mask].view(n_labels, -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        #print("BB", labels.size(), similarity_matrix.size(), mask.size(), labels.sum(1))
        """
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(n_labels, -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(n_labels, -1)

        #print("CCC", positives.size(), negatives.size())


        logits = torch.cat([positives, negatives], dim=1)
        # since the positve sample is in index 0, we set labels to zero
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z0.device)

        logits = logits / self.temperature
        return logits, labels
        """
        return similarity_matrix, labels

    def _loss(self, logits, labels):

        logits = logits / self.temperature
        logits = logits - torch.max(logits)

        n = torch.exp(logits[:, 0])
        d = torch.exp(logits[:, 1:]).sum(dim=1)
        loss = - torch.log(n / d)
        return loss.mean()
        """
        logits = torch.exp(logits)
        n = torch.sum(logits * labels)
        d = torch.sum(logits * (1 - labels))
        loss = - torch.log(n / d)
        return loss.mean()
        """

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets, cell_class_id = batch

        idx = torch.randperm(gene_ids.size(1))
        if len(idx) % 2 == 1:
            idx = idx[:-1]
        n = len(idx) // 2
        idx0 = idx[:n]
        idx1 = idx[n:]
        gene_ids_list = [gene_ids[:, idx0], gene_ids[:, idx1]]
        gene_vals_list = [gene_vals[:, idx0], gene_vals[:, idx1]]
        key_padding_mask_list = [key_padding_mask[:, idx0], key_padding_mask[:, idx1]]

        z, cell_prop_pred = self.network.forward(
            gene_ids_list, gene_vals_list, key_padding_mask_list, 2,
        )
        p_dims = [len(p["values"]) for p in self.cell_properties.values()]

        labels = torch.argmax(cell_prop_targets[:, 3, :7], dim=-1).to(torch.long)

        logits, labels = self.info_nce_loss(z[0], z[1], labels)
        contrastive_loss = self._loss(logits, labels)

        cell_prop_loss = 0.0
        if self.cell_properties is not None:
            for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
                if cell_prop["discrete"]:
                    # discrete variable, use cross entropy
                    # class values of -1 will be masked out
                    idx = torch.where(cell_prop_targets[:, n, 0] >= 0)[0]
                    if len(idx) > 0:
                        for j in range(2):
                            cell_prop_loss += self.cell_prop_cross_ent[k](
                                cell_prop_pred[j][k][idx], cell_prop_targets[idx, n, : p_dims[n]]
                            )
                else:
                    # continuous variable, use MSE
                    # class values less than -999 or greater than 999 will be masked out
                    idx = torch.where(cell_prop_targets[:, n, 0] > -999)[0]
                    if len(idx) > 0:
                        for j in range(2):
                            cell_prop_loss += self.cell_prop_mse[k](
                                cell_prop_pred[j][k][idx], cell_prop_targets[idx, n, 0]
                            )

        loss = contrastive_loss + cell_prop_loss

        self.log("cont_loss", contrastive_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("cell_loss", cell_prop_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets, cell_class_id = batch

        _, cell_prop_pred = self.network.forward(
            [gene_ids], [gene_vals], [key_padding_mask], 1,
        )
        cell_prop_pred = cell_prop_pred[0]

        p_dims = [len(p["values"]) for p in self.cell_properties.values()]

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
            if cell_prop["discrete"]:
                predict_idx = torch.argmax(cell_prop_pred[k][..., : p_dims[n]], dim=-1).to(torch.int64)
                # property values of -1 will be masked out
                idx = torch.where(cell_prop_targets[:, n, 0] >= 0)[0]
                if len(idx) > 0:
                    targets = torch.argmax(cell_prop_targets[idx, n, :], dim=-1).to(torch.int64)
                    self.cell_prop_accuracy[k].update(predict_idx[idx], targets)
                targets = torch.argmax(cell_prop_targets[:, n, :], dim=-1).to(torch.int64)
                self.results[k].append(targets.detach().cpu().numpy())
                self.results["pred_" + k].append(cell_prop_pred[k][..., : p_dims[n]].detach().to(torch.float32).cpu().numpy())
            else:
                # property values < -999  will be masked out
                idx = torch.where(cell_prop_targets[:, n, 0] > - 999)[0]
                self.cell_prop_explained_var[k].update(
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n, 0]
                )
                self.results[k].append(cell_prop_targets[:, n, 0].detach().cpu().numpy())
                self.results["pred_" + k].append(cell_prop_pred[k].detach().cpu().to(torch.float32).numpy())

        self.results["class_id"].append(cell_class_id.detach().cpu().to(torch.float32).numpy())

        for k, v in self.cell_prop_accuracy.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_prop_explained_var.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):

        v = self.trainer.logger.version
        fn = f"{self.trainer.log_dir}/lightning_logs/version_{v}/test_results.pkl"
        for k in self.cell_properties.keys():
            self.results[k] = np.stack(self.results[k])
            self.results["pred_" + k] = np.stack(self.results["pred_" + k])
        self.results["class_id"] = np.stack(self.results["class_id"])

        pickle.dump(self.results, open(fn, "wb"))

        self.results["epoch"] = self.current_epoch + 1
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []
        self.results["class_id"] = []

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
            min_lr=5e-7,
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


class AllToAllPerutbation(pl.LightningModule):
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

        # Functions and metrics
        self.mse = nn.MSELoss()

        self.cell_prop_cross_ent = None
        self.cell_prop_accuracy = None
        self.cell_prop_mse = None
        self.cell_prop_explained_var = None

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self.perturb_idx = task_cfg["perturb_idx"] if "perturb_idx" in task_cfg else None
        self.n_bins = task_cfg["n_bins"] if "n_bins" in task_cfg else None

        self._create_results_dict()
        self.source_code_copied = False

    def _create_results_dict(self):

        self.results = {"gene_names": self.gene_names}
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        N = len(self.gene_names)
        self.results["pos_delta"] = np.zeros((N, N), dtype=np.float32)
        self.results["pos_var"] = np.zeros((N, N), dtype=np.float32)
        self.results["pos_count"] = np.zeros((N, N), dtype=np.uint16)

        self.results["neg_delta"] = np.zeros((N, N), dtype=np.float32)
        self.results["neg_var"] = np.zeros((N, N), dtype=np.float32)
        self.results["neg_count"] = np.zeros((N, N), dtype=np.uint16)

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

    def training_step(self, batch, batch_idx):

        return None

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask = batch
        batch_size = gene_ids.shape[0]
        n_iters = 10

        n_input_genes = gene_vals.shape[1]

        gene_pred, _, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask,
        )

        target_idx = gene_target_ids.to(torch.int16).detach().cpu().numpy()
        rnd_idx = np.random.choice(n_input_genes, size=(10,), replace=False)

        for n in range(n_iters):

            rnd_inc = torch.from_numpy(np.random.choice([1, 2, 3], size=(batch_size,), replace=True)).to(gene_vals.device)
            gene_vals_perturb = copy.deepcopy(gene_vals.clone().detach())
            # print("MEAN GEN VALS", gene_vals.mean())
            gene_vals_perturb[:, rnd_idx[n]] = torch.clip(
                gene_vals_perturb[:, rnd_idx[n]] + rnd_inc, 0, self.n_bins-1
            )

            gene_pred_perturb, _, _ = self.network.forward(
                gene_ids, gene_target_ids, gene_vals_perturb, key_padding_mask,
            )

            source_idx = gene_ids[:, rnd_idx[n]].to(torch.int16).detach().cpu().numpy()
            delta = gene_pred_perturb.to(torch.float32).detach().cpu().numpy() - gene_pred.to(torch.float32).detach().cpu().numpy()
            delta = delta[:, :, 0]
            #print("A", delta.std(), torch.sum(torch.abs(gene_vals_perturb - gene_vals)))

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


    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        fn = f"{self.trainer.log_dir}/test_results.pkl"
        pickle.dump(self.results, open(fn, "wb"))

    def configure_optimizers(self):

        return torch.optim.SGD(
            self.network.parameters(),
            lr=0.0,
        )


class MSELoss(pl.LightningModule):
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

        # Functions and metrics
        self.mse = nn.MSELoss()
        self.val_mse = MeanSquaredError()

        self.cell_prop_cross_ent = None
        self.cell_prop_accuracy = None
        self.cell_prop_mse = None
        self.cell_prop_explained_var = None

        self.gene_names = task_cfg["gene_names"] if "gene_names" in task_cfg else None
        self.perturb_idx = task_cfg["perturb_idx"] if "perturb_idx" in task_cfg else None
        self.n_bins = task_cfg["n_bins"] if "n_bins" in task_cfg else None

        self.gene_explained_var = ExplainedVariance()

        self._create_results_dict()
        self.source_code_copied = False

    def _create_results_dict(self):

        self.results = {"epoch": 0, "gene_names": self.gene_names}
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        N = len(self.gene_names)
        self.results["target_gene_counts"] = np.zeros(N)
        self.results["target_gene_preds"] = np.zeros(N)
        self.results["atac_data"] = []
        self.results["pred_atac_data"] = []

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

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask = batch
        gene_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask,
        )
        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))

        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return gene_loss


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, gene_vals, gene_targets, key_padding_mask = batch
        """
        if self.perturb_idx is not None:
            perturb_val = self.current_epoch
            idx = torch.where(gene_ids == self.perturb_idx)
            for i, j in zip(idx[0], idx[1]):
                gene_vals[i, j] = torch.clip(
                    gene_vals[i, j] + perturb_val, 0, self.n_bins-1
                )
        """

        gene_pred, _ = self.network.forward(
            gene_ids, gene_target_ids, gene_vals, key_padding_mask,
        )

        for i in range(gene_pred.shape[0]):
            idx = gene_target_ids[i, :].detach().cpu().numpy()
            y = np.squeeze(gene_pred[i, :].to(torch.float32).detach().cpu().numpy())
            self.results["target_gene_counts"][idx] += 1
            self.results["target_gene_preds"][idx] += y

        #self.results["gene_target_ids"].append(gene_target_ids.to(torch.int16).detach().cpu().numpy())
        #self.results["gene_vals"].append(gene_vals.detach().cpu().numpy())
        #self.results["gene_ids"].append(gene_ids.detach().cpu().numpy())
        #self.results["gene_targets"].append(gene_targets.detach().cpu().numpy())
        #self.results["gene_pred"].append(gene_pred.to(torch.float32).detach().cpu().numpy())

        self.gene_explained_var(gene_pred, gene_targets.unsqueeze(2))
        self.val_mse(gene_pred, gene_targets.unsqueeze(2))
        self.log("exp_var", self.gene_explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
            self.log("learn_rate", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            break


    def on_validation_epoch_end(self):

        if not self.source_code_copied:
            self._copy_source_code()
            self.source_code_copied = True

        if self.perturb_idx is None:
            fn = f"{self.trainer.log_dir}/test_results.pkl"
        else:
            fn = f"{self.trainer.log_dir}/test_results_ep{self.current_epoch}.pkl"
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    if len(self.results[k]) > 0:
        #        self.results[k] = np.concatenate(self.results[k], axis=0)
        pickle.dump(self.results, open(fn, "wb"))
        self.results["epoch"] = self.current_epoch + 1
        #for k in ["gene_target_ids", "gene_targets", "gene_vals", "gene_ids", "gene_pred"]:
        #    self.results[k] = []
        N = len(self.gene_names)
        #self.results["target_gene_counts"] = np.zeros(N)
        #self.results["target_gene_preds"] = np.zeros(N)



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
