import pickle
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics.classification import Accuracy


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
        self.cell_properties = self.network.cell_properties

        # Functions and metrics
        self.mse = nn.MSELoss()
        if self.cell_properties is not None:
            self.cell_prop_cross_ent = nn.ModuleDict()
            self.cell_prop_mse = nn.ModuleDict()
            self.cell_prop_accuracy = nn.ModuleDict()
            self.cell_prop_explained_var = nn.ModuleDict()

            for k, v in self.network.cell_properties.items():
                if v is not None:
                    # categorical variable
                    weight = torch.from_numpy(np.float32(1 / v)) if task_cfg["balance_classes"] else None
                    self.cell_prop_cross_ent[k] = nn.CrossEntropyLoss(weight=weight)
                    self.cell_prop_accuracy[k] = Accuracy(task="multiclass", num_classes=len(v), average="macro")
                else:
                    # continuous variable
                    self.cell_prop_mse[k] = nn.MSELoss()
                    self.cell_prop_explained_var[k] = ExplainedVariance()
        else:
            self.cell_prop_cross_ent = self.cell_prop_accuracy = self.cell_prop_mse = self.cell_prop_explained_var = None

        self.gene_explained_var = ExplainedVariance()
        self.metrics = MetricCollection([ExplainedVariance()])
        self.train_metrics = self.metrics.clone(prefix="train_")
        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")

        self._create_results_dict()

    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.network.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets = batch
        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, key_padding_mask,
        )

        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        cell_prop_loss = 0

        if self.network.cell_properties is not None:
            for n, (k, v) in enumerate(self.network.cell_properties.items()):
                if k in self.cell_prop_cross_ent.keys():
                    # class values of -1 will be masked out
                    idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                    cell_prop_loss += self.cell_prop_cross_ent[k](
                        cell_prop_pred[k][idx], cell_prop_targets[idx, n].to(torch.int64)
                    )
                elif k in self.cell_prop_mse.keys():
                    # class values less than -999 or greater than 999 will be masked out
                    idx = torch.where(cell_prop_targets[:, n] > -999)[0]
                    cell_prop_loss += self.cell_prop_mse[k](
                        cell_prop_pred[k][idx], cell_prop_targets[idx, n]
                    )

        # TODO: fit this
        alpha = 1.0
        beta = 1.0
        loss = alpha * gene_loss + beta * cell_prop_loss

        self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("cell_loss", cell_prop_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets = batch

        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, key_padding_mask,
        )

        self.gene_explained_var(gene_pred, gene_targets.unsqueeze(2))

        if self.network.cell_properties is not None:
            for n, (k, v) in enumerate(self.network.cell_properties.items()):
                if k in self.cell_prop_cross_ent.keys():
                    predict_idx = torch.argmax(cell_prop_pred[k], dim=-1)
                    # property values of -1 will be masked out
                    idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                    self.cell_prop_accuracy[k].update(
                        predict_idx[idx], cell_prop_targets[idx, n]
                    )
                    self.results[k].append(cell_prop_targets[:, n].detach().cpu().numpy())
                    self.results["pred_" + k].append(predict_idx.detach().cpu().numpy())
                elif k in self.cell_prop_mse.keys():
                    # property values < -999  will be masked out
                    idx = torch.where(cell_prop_targets[:, n] > - 999)[0]
                    self.cell_prop_explained_var[k].update(
                        cell_prop_pred[k][idx], cell_prop_targets[idx, n]
                    )
                    self.results[k].append(cell_prop_targets[:, n].detach().cpu().numpy())
                    self.results["pred_" + k].append(cell_prop_pred[k][idx].detach().cpu().to(torch.float32).numpy())

        self.log("gene_ex", self.gene_explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k, v in self.cell_prop_accuracy.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.cell_prop_explained_var.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self):

        v = self.trainer.logger.version
        fn = f"{self.trainer.log_dir}/lightning_logs/version_{v}/test_results.pkl"
        for k in self.network.cell_properties.keys():
            self.results[k] = np.stack(self.results[k])
            self.results["pred_" + k] = np.stack(self.results["pred_" + k])

        pickle.dump(self.results, open(fn, "wb"))

        self.results["epoch"] = self.current_epoch + 1
        for k in self.network.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []

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


class Classification(MSELoss):
    def __init__(
        self,
        network,
        task_cfg,
        **kwargs
    ):

        # Initialize superclass
        super().__init__(None, task_cfg)
        self.network = network
        self.task_cfg = task_cfg

        # Functions and metrics
        self.cross_ent = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=task_cfg["n_classes"], average="macro")
        self.metrics = MetricCollection([ExplainedVariance()])
        self.train_metrics = self.metrics.clone(prefix="train_")
        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")

    def training_step(self, batch, batch_idx):

        gene_ids, gene_vals, key_padding_mask, classes = batch
        decoder_out, latent = self.network.forward(
            gene_ids, gene_vals, key_padding_mask,
        )

        y_hat = self.network.mlp(decoder_out).squeeze()

        #target = nn.functional.one_hot(classes, num_classes=self.task_cfg["n_classes"])
        target = classes.to(dtype=torch.int64)
        loss = self.cross_ent(y_hat, target)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        gene_ids, gene_vals, key_padding_mask, classes = batch
        decoder_out, latent = self.network.forward(
            gene_ids, gene_vals, key_padding_mask,
        )
        y_hat = self.network.mlp(decoder_out).squeeze()

        #target = nn.functional.one_hot(classes, num_classes=self.task_cfg["n_classes"])
        target = classes.to(dtype=torch.int64)

        loss = self.cross_ent(y_hat, target)
        acc = self.accuracy(y_hat, target)

        self.log("accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

