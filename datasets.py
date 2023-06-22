from typing import Dict, List, Optional

import os
import torch
import numpy as np
import cloudpickle as pickle
import pytorch_lightning as pl
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    BatchSampler,
    SequentialSampler,
)

class SingleCellDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        metadata_path: str,
        cells_per_epochs: int,
        predict_classes: Optional[Dict[str, int]] = None,
        n_mask: int = 316,
        batch_size: int = 32,
        rank_order: bool = False,
        scale_by_max: bool = True,
        pin_memory: bool = False,
    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))
        self.data_path = data_path
        self.cells_per_epochs = cells_per_epochs
        self.predict_classes = predict_classes
        self.n_samples = len(self.metadata["obs"]["class"])
        self.n_genes = len(self.metadata["var"]["gene_name"])
        self.n_classes = len(predict_classes) if predict_classes is not None else 0
        self.n_mask = n_mask
        self.batch_size = batch_size
        self.rank_order = rank_order
        self.scale_by_max = scale_by_max
        self.pin_memory = pin_memory

        self.offset = 2 * self.n_genes  # FP16 is 2 bytes
        self._get_class_info()
        self._create_gene_class_ids()


    def __len__(self):
        return self.n_samples

    def _create_gene_class_ids(self):
        """"Create the gene and class ids. Will start with the gene ids, and then concatenate
        the class ids if requested"""
        gene_ids = torch.arange(0, self.n_genes).repeat(self.batch_size, 1)
        if self.predict_classes is not None:
            class_ids = torch.arange(0, self.n_classes).repeat(self.batch_size, 1)
        else:
            class_ids = None

        return gene_ids, class_ids

    def _get_class_info(self):
        """Extract the list of uniques values for each class (e.g. sex, cell type, etc.) to be predicted"""
        if self.predict_classes is not None:
            self.class_unique = {}
            self.class_dist = {}
            for k in self.predict_classes.keys():
                unique_list, counts = np.unique(self.metadata["obs"][k], return_counts=True)
                print("unique_list", unique_list)
                print("unique_count", counts)
                self.class_unique[k] = np.array(unique_list)
                self.class_dist[k] = counts / np.max(counts)
        else:
            self.class_unique = self.class_dist = None

    def _get_class_vals(self, idx: List[int]):
        """Extract the class value for ach entry in the batch"""
        if self.class_unique is None:
            return None

        class_vals = np.zeros((self.batch_size, self.n_classes), dtype=np.int64)
        for n0, i in enumerate(idx):
            for n1, (k, v) in enumerate(self.class_unique.items()):
                class_vals[n0, n1] = np.where(self.metadata["obs"][k][i] == v)[0]

        return torch.from_numpy(class_vals)

    def _get_gene_vals(self, idx: List[int]):

        gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        for n, i in enumerate(idx):
            gene_vals[n, :] = np.memmap(
                self.data_path, dtype='float16', mode='r', shape=(self.n_genes,), offset=i * self.offset
            ).astype(np.float32)
            if self.scale_by_max:
                gene_vals[n, :] /= (1e-9 + self.metadata["stats"]["max"])

        zero_idx = np.where(gene_vals == 0)
        gene_vals = torch.from_numpy(gene_vals)
        # return two copies since we'll modify gene_vals but keep gene_targets as is
        return gene_vals, gene_vals, zero_idx

    def _prepare_data(self, batch_idx):

        gene_vals, gene_targets, zero_idx = self._get_gene_vals(batch_idx)
        class_vals = self._get_class_vals(batch_idx)

        key_padding_mask = torch.zeros_like(gene_vals).detach()
        key_padding_mask[zero_idx[0], zero_idx[1]] = 1.0

        return gene_vals, key_padding_mask, gene_targets, class_vals

    def __len__(self):
        return self.cells_per_epochs

    def __getitem__(self, idx: List[int]):

        if len(idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        gene_vals, key_padding_mask, gene_targets, class_targets = self._prepare_data(idx)

        # mask indices
        rnd_idx = []
        for i in range(self.batch_size):
            nonzero_idx = torch.where(key_padding_mask[i, :] == 0)[0]
            k = np.random.choice(nonzero_idx, self.n_mask, replace=False)
            rnd_idx.append(k)
        rnd_idx = np.concatenate(rnd_idx, axis=-1)
        mask_col_ids = torch.tensor(rnd_idx).long()
        mask_row_ids = torch.repeat_interleave(
            torch.arange(0, self.batch_size), self.n_mask
        ).long()

        assert len(mask_col_ids) == len(mask_row_ids)

        # the genes to predict will be masked out in the input
        gene_targets = gene_targets[mask_row_ids, mask_col_ids].reshape(self.batch_size, -1)

        gene_ids, class_ids = self._create_gene_class_ids()

        # target ids are the genes that are masked out plus the classes to predict
        gene_target_ids = gene_ids[mask_row_ids, mask_col_ids].reshape(self.batch_size, -1)

        #if not self.rank_order:
        # for targets, mask out gene values and assign their index to the padding_index
        gene_ids[mask_row_ids, mask_col_ids] = self.n_genes
        gene_vals[mask_row_ids, mask_col_ids] = 0.0

        # mask out all genes with expression of zero
        for i in range(self.batch_size):
            zero_idx = torch.where(key_padding_mask[i, :] == 1)[0]
            gene_ids[i, zero_idx] = self.n_genes
            gene_vals[i, zero_idx] = 0.0 #  should already be zero...just to be safe


        batch = gene_ids, gene_target_ids, class_ids, gene_vals, gene_targets, key_padding_mask, class_targets

        if self.pin_memory:
            for tensor in batch:
                tensor.pin_memory()

        return batch


class DataModule(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        data_path: str,
        frac: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 16,
        n_mask: int = 316,
        rank_order: bool = False,
        predict_classes: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.frac = frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_mask = n_mask
        self.rank_order = rank_order
        self.predict_classes = predict_classes

    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            os.path.join(self.data_path, "train_data.dat"),
            os.path.join(self.data_path, "train_metadata.pkl"),
            32 * 2000,
            predict_classes=self.predict_classes,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
        )
        self.val_dataset = SingleCellDataset(
            os.path.join(self.data_path, "test_data.dat"),
            os.path.join(self.data_path, "test_metadata.pkl"),
            32 * 200,
            predict_classes=self.predict_classes,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
        )
        self.n_genes = self.train_dataset.n_genes
        print(f"number of genes {self.n_genes}")

    # return the dataloader for each split
    def train_dataloader(self):
        sampler = BatchSampler(
            RandomSampler(self.train_dataset),
            batch_size=self.train_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.train_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return dl

    def val_dataloader(self):
        sampler = BatchSampler(
            SequentialSampler(self.val_dataset),
            batch_size=self.val_dataset.batch_size,
            drop_last=True,
        )
        dl = DataLoader(
            self.val_dataset,
            batch_size=None,
            batch_sampler=None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return dl

