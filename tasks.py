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
        self.class_dist = self.network.class_dist

        # Functions and metrics
        self.mse = nn.MSELoss()
        if self.class_dist is not None:
            self.cross_ent = {}
            self.accuracy = {}
            for k, v in self.network.class_dist.items():
                self.cross_ent[k] = nn.CrossEntropyLoss(weight=torch.from_numpy(np.float32(1 / v)))
                self.accuracy[k] = Accuracy(task="multiclass", num_classes=len(v), average="macro")
        else:
            self.cross_ent = self.accuracy = None

        self.explained_var = ExplainedVariance()
        self.metrics = MetricCollection([ExplainedVariance()])
        self.train_metrics = self.metrics.clone(prefix="train_")
        self.val_metrics = self.metrics.clone(prefix="val_")
        self.test_metrics = self.metrics.clone(prefix="test_")

        self._create_results_dict()

    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.network.class_dist.keys():
            self.results[k] = []
            self.results["pred_" + k] = []

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, class_ids, gene_vals, gene_targets, key_padding_mask, class_targets = batch
        gene_pred, class_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, class_ids, gene_vals, key_padding_mask,
        )

        mse_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        class_loss = 0

        if self.network.class_dist is not None:
            for n, (k, v) in enumerate(self.network.class_dist.items()):
                cross_entropy = self.cross_ent[k].to(device=class_pred[k].device)
                # class values of -1 will be masked out
                idx = torch.where(class_targets[:, n] >= 0)[0]
                class_loss += cross_entropy(class_pred[k][idx], class_targets[idx, n])

        # TODO: fit this
        alpha = 10.0
        beta = 1.0
        loss = alpha * mse_loss + beta * class_loss

        self.log("train_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_class", class_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, class_ids, gene_vals, gene_targets, key_padding_mask, class_targets = batch

        gene_pred, class_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, class_ids, gene_vals, key_padding_mask,
        )
        loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        ev = self.explained_var(gene_pred, gene_targets.unsqueeze(2))

        if self.network.class_dist is not None:
            acc = {}

            for n, k in enumerate(self.network.class_dist.keys()):
                class_predict_idx = torch.argmax(class_pred[k], dim=-1)
                metric = self.accuracy[k].to(device=class_pred[k].device)
                # class values of -1 will be masked out
                idx = torch.where(class_targets[:, n] >= 0)[0]
                acc[k] = metric(class_pred[k][idx], class_targets[idx, n])
                self.results[k].append(class_targets[idx, n].detach().cpu().numpy())
                self.results["pred_" + k].append(class_predict_idx[idx].detach().cpu().numpy())

        self.log("gene_ex", ev, on_step=False, on_epoch=True, prog_bar=True)

        for k, v in acc.items():
            self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):

        v = self.trainer.logger.version
        fn = f"{self.trainer.log_dir}/lightning_logs/version_{v}/test_results.pkl"
        for k in self.network.class_dist.keys():
            self.results[k] = np.stack(self.results[k])
            self.results["pred_" + k] = np.stack(self.results["pred_" + k])

        pickle.dump(self.results, open(fn, "wb"))

        self.results["epoch"] = self.current_epoch + 1
        for k in self.network.class_dist.keys():
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

