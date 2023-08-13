import pickle
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from torchmetrics import MetricCollection, ExplainedVariance
from torchmetrics.classification import Accuracy
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
        self.variational = task_cfg["variational"]

        if self.variational:
            self.prior = Normal(
                torch.zeros(1, task_cfg["bottleneck_dim"]).to(torch.device("cuda")),
                torch.ones(1, task_cfg["bottleneck_dim"]).to(torch.device("cuda")),
            )

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
                        np.sqrt(np.float32(np.clip(1 / cell_prop["freq"], 0.1, 10.0)))
                    ) if task_cfg["balance_classes"] else None
                    self.cell_prop_cross_ent[k] = nn.CrossEntropyLoss(weight=weight)
                    #self.cell_prop_cross_ent[k] = FocalLoss(gamma = 2.0)
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
            self.results["class_id"] = []

    def training_step(self, batch, batch_idx):

        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets, cell_class_id = batch
        gene_pred, cell_prop_pred, latent = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, key_padding_mask, training=True,
        )
        p_dims = [len(p["values"]) for p in self.cell_properties.values()]

        # gene_loss = self.mse(gene_pred, gene_targets.unsqueeze(2))
        gene_loss = 0.0
        cell_prop_loss = 0.0

        if self.variational:
            beta = 1e-7 * np.clip((self.current_epoch) / 20.0, 0.0, 1.0)
            enc_mean, enc_std = latent
            enc_dist = Normal(enc_mean, enc_std)
            kl_vec = torch.distributions.kl.kl_divergence(enc_dist, self.prior)
            kl_loss = kl_vec.mean()
            self.log("KL", kl_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("beta", beta, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.cell_properties is not None:
            for n, (k, cell_prop) in enumerate(self.cell_properties.items()):
                if cell_prop["discrete"]:
                    # discrete variable, use cross entropy
                    # class values of -1 will be masked out
                    idx = torch.where(cell_prop_targets[:, n, 0] >= 0)[0]
                    if len(idx) > 0:
                        cell_prop_loss += self.cell_prop_cross_ent[k](
                            cell_prop_pred[k][idx], cell_prop_targets[idx, n, : p_dims[n]]
                        )
                else:
                    # continuous variable, use MSE
                    # class values less than -999 or greater than 999 will be masked out
                    idx = torch.where(cell_prop_targets[:, n, 0] > -999)[0]
                    if len(idx) > 0:
                        cell_prop_loss += self.cell_prop_mse[k](
                            cell_prop_pred[k][idx], cell_prop_targets[idx, n, 0]
                        )

        loss = self.task_cfg["gene_weight"] * gene_loss + self.task_cfg["cell_prop_weight"] * cell_prop_loss
        if self.variational:
            loss = loss + beta * kl_loss

        # self.log("gene_loss", gene_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("cell_loss", cell_prop_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):


        gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_targets, cell_class_id = batch
        gene_pred, cell_prop_pred, _ = self.network.forward(
            gene_ids, gene_target_ids, cell_prop_ids, gene_vals, key_padding_mask, training=False,
        )

        p_dims = [len(p["values"]) for p in self.cell_properties.values()]
        # self.gene_explained_var(gene_pred, gene_targets.unsqueeze(2))

        if self.cell_properties is not None:
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

        # self.log("gene_ex", self.gene_explained_var, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
