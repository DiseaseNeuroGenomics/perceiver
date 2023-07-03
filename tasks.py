import pickle
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
        self.cell_properties = self.task_cfg["cell_properties"]

        # Functions and metrics
        self.mse = nn.MSELoss()
        if self.cell_properties is not None:
            self.cell_prop_cross_ent = nn.ModuleDict()
            self.cell_prop_mse = nn.ModuleDict()
            self.cell_prop_accuracy = nn.ModuleDict()
            self.cell_prop_explained_var = nn.ModuleDict()

            for k, cell_prop in self.cell_properties.items():
                if cell_prop["discrete"]:
                    # discrete variable, set up cross entropy module
                    weight = torch.from_numpy(
                        np.float32(np.minimum(1 / cell_prop["freq"], 25.0))
                    ) if task_cfg["balance_classes"] else None
                    self.cell_prop_cross_ent[k] = nn.CrossEntropyLoss(weight=weight)
                    self.cell_prop_accuracy[k] = Accuracy(
                        task="multiclass", num_classes=len(cell_prop["values"]), average="macro",
                    )
                else:
                    # continuous variable, set up MSE module
                    self.cell_prop_mse[k] = nn.MSELoss()
                    self.cell_prop_explained_var[k] = ExplainedVariance()
        else:
            self.cell_prop_cross_ent = None
            self.cell_prop_accuracy = None
            self.cell_prop_mse = None
            self.cell_prop_explained_var = None

        self.gene_explained_var = ExplainedVariance()
        self.metrics = MetricCollection([ExplainedVariance()])

        self._create_results_dict()


    def _create_results_dict(self):

        self.results = {"epoch": 0}
        for k in self.cell_properties.keys():
            self.results[k] = []
            self.results["pred_" + k] = []

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets = batch
        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, key_padding_mask,
        )

        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        cell_prop_loss = 0

        if self.cell_properties is not None:
            for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
                if cell_prop["discrete"]:
                    # discrete variable, use cross entropy
                    # class values of -1 will be masked out
                    idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                    cell_prop_loss += self.cell_prop_cross_ent[k](
                        cell_prop_pred[k][idx], cell_prop_targets[idx, n].to(torch.int64)
                    )
                else:
                    # continuous variable, use MSE
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

        if self.cell_properties is not None:
            for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
                if cell_prop["discrete"]:
                    predict_idx = torch.argmax(cell_prop_pred[k], dim=-1)
                    # property values of -1 will be masked out
                    idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                    if len(idx) > 0:
                        self.cell_prop_accuracy[k].update(
                            predict_idx[idx], cell_prop_targets[idx, n]
                        )
                    self.results[k].append(cell_prop_targets[:, n].detach().cpu().numpy())
                    self.results["pred_" + k].append(predict_idx.detach().cpu().numpy())
                else:
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
        for k in self.cell_properties.keys():
            self.results[k] = np.stack(self.results[k])
            self.results["pred_" + k] = np.stack(self.results["pred_" + k])

        pickle.dump(self.results, open(fn, "wb"))

        self.results["epoch"] = self.current_epoch + 1
        for k in self.cell_properties.keys():
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


class AdverserialLoss(MSELoss):
    def __init__(
        self,
        network,
        task_cfg,
        adv_cell_prop: str = "SubID",
        adv_loss_ratio: float = 0.025,
        **kwargs
    ):
        # Initialize superclass
        super().__init__(network, task_cfg, **kwargs)
        print("ADV LOSS COEFF", adv_loss_ratio)
        self.automatic_optimization = False
        self.network_params = [p for n, p in self.network.named_parameters() if "SubID" not in n]
        self.adv_params = [p for n, p in self.network.named_parameters() if "SubID" in n]
        self.adv_cell_prop = adv_cell_prop
        self.adv_loss_ratio = adv_loss_ratio

        self.adv_accuracy = nn.ModuleDict()
        self.adv_accuracy[adv_cell_prop] = Accuracy(
            task="multiclass", num_classes=len(self.cell_properties[adv_cell_prop]["values"]), average="macro",
        )
        self.steps = 0.0


    def training_step(self, batch, batch_idx):

        opt0, opt1 = self.configure_optimizers()
        opt0, opt1 = self.update_lr(opt0, opt1)
        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets = batch
        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, key_padding_mask,
        )

        gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        cell_prop_loss = 0

        for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
            alpha = - self.adv_loss_ratio if k == self.adv_cell_prop else 1.0
            if cell_prop["discrete"]:
                # discrete variable, use cross entropy
                # class values of -1 will be masked out
                idx = torch.where(cell_prop_targets[:, n] >= 0)[0]
                cell_prop_loss += alpha * self.cell_prop_cross_ent[k](
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n].to(torch.int64)
                )
                if k == self.adv_cell_prop:
                    opt1.zero_grad()
                    adv_loss = self.cell_prop_cross_ent[self.adv_cell_prop](
                        cell_prop_pred[self.adv_cell_prop][idx], cell_prop_targets[idx, n].to(torch.int64)
                    )
                    self.manual_backward(adv_loss, retain_graph=True)
                    #opt1.step()
                    # TODO: currently not displaying accuracy, only the loss
                    self.adv_accuracy[self.adv_cell_prop].update(cell_prop_pred[k][idx], cell_prop_targets[idx, n])
            else:
                # continuous variable, use MSE
                # class values less than -999 or greater than 999 will be masked out
                idx = torch.where(cell_prop_targets[:, n] > -999)[0]
                cell_prop_loss += alpha * self.cell_prop_mse[k](
                    cell_prop_pred[k][idx], cell_prop_targets[idx, n]
                )

        # TODO: fit this
        alpha = 1.0
        beta = 1.0
        loss = alpha * gene_loss + beta * cell_prop_loss

        opt0.zero_grad()
        self.manual_backward(loss)

        opt0.step()
        opt1.step()

       #  self.update_lr(opt0, opt1)

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

        return loss

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

