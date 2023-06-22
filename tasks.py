import torch
from torch import nn
import copy
import numpy as np
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics.classification import MulticlassAccuracy
from datasets import DataModule


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
        self.class_dist = self.network.class_dist

        # Functions and metrics
        self.mse = nn.MSELoss()
        if self.class_dist is not None:
            self.cross_ent = {}
            self.accuracy = {}
            for k, v in self.class_dist.items():
                print(f"Class weights {1/v}")
                self.cross_ent[k] = nn.CrossEntropyLoss(weight=torch.from_numpy(np.float32(1 / v)))
                #self.cross_ent[k] = nn.CrossEntropyLoss(weight=torch.from_numpy(np.ones_like(v)))
                #self.cross_ent[k] = nn.CrossEntropyLoss()
                self.accuracy[k] = MulticlassAccuracy(num_classes=len(v), average="macro")
        else:
            self.cross_ent = self.accuracy = None

        self.explained_var = ExplainedVariance()
        self.metrics = MetricCollection([ExplainedVariance()])
        self.train_metrics = self.metrics.clone(prefix="train_")
        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")


    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, class_ids, gene_vals, gene_targets, key_padding_mask, class_targets = batch
        gene_pred, class_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, class_ids, gene_vals, key_padding_mask,
        )

        mse_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        class_loss = 0

        if self.class_dist is not None:
            for n, (k, v) in enumerate(self.class_dist.items()):
                cross_entropy = self.cross_ent[k].to(device=class_pred[k].device)
                class_loss += cross_entropy(class_pred[k], class_targets[:, n])

        alpha = 10.0
        beta = 1.0
        loss = alpha * mse_loss + beta * class_loss


        self.log("train_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_class", class_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, class_ids, gene_vals, gene_targets, key_padding_mask, class_targets = batch

        gene_pred, class_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, class_ids, gene_vals, key_padding_mask,
        )
        loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        ev = self.explained_var(gene_pred, gene_targets.unsqueeze(2))

        if self.class_dist is not None:
            acc = {}

            for n, k in enumerate(self.class_dist.keys()):
                class_predict_idx = torch.argmax(class_pred[k], dim=-1)#.to(device="cpu")
                metric = self.accuracy[k].to(device=class_pred[k].device)
                acc[k] = metric(class_predict_idx, class_targets[:, n])
                # print(class_targets[:, n])
        # metrics = self.val_metrics(y_hat, mask_vals.unsqueeze(2))
        # loss_dict = {"val_MSELoss": loss}
        self.log("val_EV", ev, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in acc.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("tissue_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        gene_ids, gene_vals, mask_ids, mask_vals, key_padding_mask = batch

        decoder_out = self.network.forward(
            gene_ids, gene_vals, gene_ids, key_padding_mask, mask_ids
        )
        y_hat = self.network.mlp(decoder_out)
        return y_hat, mask_vals


    def test_step_end(self, results):
        y_hat, mask_vals = results
        loss = self.mse(y_hat, mask_vals.unsqueeze(2))
        metrics = self.test_metrics(y_hat, mask_vals.unsqueeze(2))
        loss_dict = {"test_MSELoss": loss}
        self.log_dict(loss_dict | metrics, on_step=True, on_epoch=True, sync_dist=True)


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
            optimizer_idx,
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

