from typing import Any, Dict, List, Optional

import os
import torch
import numpy as np
import scanpy as sc
import pickle
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
        cell_properties: Optional[Dict[str, Any]] = None,
        n_mask: int = 100,
        batch_size: int = 32,
        normalize_total: Optional[float] = 10_000,
        log_normalize: bool = True,
        rank_order: bool = False,
        scale_by_max: bool = False,
        pin_memory: bool = False,
        same_latent_class: bool = True,
        max_cell_prop_val: float = 999,
    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))
        self.data_path = data_path
        self.cell_properties = cell_properties
        self.n_samples = len(self.metadata["obs"]["class"])
        print(f"Number of cells {self.n_samples}")
        self.n_genes = len(self.metadata["var"])
        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0
        self.n_mask = n_mask
        self.batch_size = batch_size
        if rank_order:
            print("Since rank_oder=True, setting normalize_total to None and log_normalize to False")
            normalize_total = None
            log_normalize = False
        if normalize_total or log_normalize:
            print("Since log_normalize or normalize_total=True, setting rank_order and scale_by_max to False")
            scale_by_max = False
            rank_order = False

        self.normalize_total = normalize_total
        self.log_normalize = log_normalize
        self.rank_order = rank_order
        self.scale_by_max = scale_by_max
        self.pin_memory = pin_memory
        self.same_latent_class = same_latent_class
        self.max_cell_prop_val = max_cell_prop_val

        self.offset = 2 * self.n_genes  # FP16 is 2 bytes
        self._get_cell_prop_info()
        self._create_gene_cell_prop_ids()


    def __len__(self):
        return self.n_samples

    def _create_gene_cell_prop_ids(self):
        """"Create the gene and class ids. Will start with the gene ids, and then concatenate
        the cell property ids if requested"""
        gene_ids = torch.arange(0, self.n_genes).repeat(self.batch_size, 1)
        if self.n_cell_properties > 0:
            if self.same_latent_class:
                # this will project all the class related latent info onto the same subspace, simplifying analysis
                cell_prop_ids = torch.zeros((self.batch_size, self.n_cell_properties), dtype=torch.int64)
            else:
                cell_prop_ids = torch.arange(0, self.n_cell_properties).repeat(self.batch_size, 1)
        else:
            cell_prop_ids = None

        return gene_ids, cell_prop_ids

    def _get_cell_prop_info(self):
        """Extract the list of uniques values for each cell property (e.g. sex, cell type, etc.) to be predicted"""
        if self.n_cell_properties > 0:
            for k, cell_prop in self.cell_properties.items():

                if not cell_prop["discrete"]:
                    # for cell properties with continuous value, determine the mean/std for normalization
                    cell_vals = self.metadata["obs"][k]
                    # remove nans, negative values, or anything else suspicious
                    idx = [n for n, cv in enumerate(cell_vals) if cv >= 0 and cv < self.max_cell_prop_val]
                    self.cell_properties[k]["mean"] = np.mean(cell_vals[idx])
                    self.cell_properties[k]["std"] = np.std(cell_vals[idx])

                elif cell_prop["discrete"] and cell_prop["values"] is None:
                    # for cell properties with discrete value, determine the possible values if none were supplied
                    # and find their distribution
                    unique_list, counts = np.unique(self.metadata["obs"][k], return_counts=True)
                    # remove nans, negative values, or anything else suspicious
                    idx = [
                        n for n, u in enumerate(unique_list) if (
                            isinstance(u, str) or np.abs(u) < self.max_cell_prop_val
                        )
                    ]
                    self.cell_properties[k]["values"] = unique_list[idx]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])

                elif cell_prop["discrete"] and cell_prop["values"] is not None:
                    unique_list, counts = np.unique(self.metadata["obs"][k], return_counts=True)
                    idx = [n for n, u in enumerate(unique_list) if u in cell_prop["values"]]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])

                print("Cell property info", k, self.cell_properties[k])
        else:
            self.cell_prop_dist = None

    def _get_cell_prop_vals(self, batch_idx: List[int]):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        cell_prop_vals = np.zeros((self.batch_size, self.n_cell_properties), dtype=np.float32)
        for n0, i in enumerate(batch_idx):
            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                if not cell_prop["discrete"]:
                    # continuous value
                    cell_val = self.metadata["obs"][k][i]
                    if np.abs(cell_val) > self.max_cell_prop_val:
                        cell_prop_vals[n0, n1] = -9999
                    else:
                        # normalize
                        cell_prop_vals[n0, n1] = (cell_val - cell_prop["mean"]) / cell_prop["std"]
                else:
                    # discrete value
                    idx = np.where(self.metadata["obs"][k][i] == np.array(cell_prop["values"]))[0]
                    # cell property values of -1 will imply N/A, and will be masked out
                    cell_prop_vals[n0, n1] = -1 if len(idx) == 0 else idx[0]

        return torch.from_numpy(cell_prop_vals)

    def _get_gene_vals(self, batch_idx: List[int]):

        gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        for n, i in enumerate(batch_idx):
            gene_vals[n, :] = np.memmap(
                self.data_path, dtype='float16', mode='r', shape=(self.n_genes,), offset=i * self.offset
            ).astype(np.float32)

            if self.scale_by_max:
                gene_vals[n, :] /= (1e-9 + self.metadata["stats"]["max"])
            elif self.normalize_total or self.log_normalize:
                gene_vals[n, :] = self._normalize(gene_vals[n, :])

        zero_idx = np.where(gene_vals == 0)
        gene_vals = torch.from_numpy(gene_vals)
        # return two copies since we'll modify gene_vals but keep gene_targets as is
        return gene_vals, gene_vals, zero_idx

    def _normalize(self, x: np.ndarray) -> np.ndarray:

        x = x * self.normalize_total / np.sum(x) if self.normalize_total is not None else x
        x = np.log1p(x) if self.log_normalize else x
        return x

    def _prepare_data(self, batch_idx):

        gene_vals, gene_targets, zero_idx = self._get_gene_vals(batch_idx)
        cell_prop_vals = self._get_cell_prop_vals(batch_idx)

        key_padding_mask = torch.zeros_like(gene_vals).detach()
        key_padding_mask[zero_idx[0], zero_idx[1]] = 1.0

        return gene_vals, key_padding_mask, gene_targets, cell_prop_vals

    def __getitem__(self, idx: List[int]):

        if len(idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        gene_vals, key_padding_mask, gene_targets, cell_prop_vals = self._prepare_data(idx)

        # mask indices
        mask_col_ids = []
        mask_row_ids = []
        for i in range(self.batch_size):
            nonzero_idx = torch.where(key_padding_mask[i, :] == 0)[0]
            # randomly choose number to mask for each cell
            # n = np.random.randint(0, self.n_mask)
            mask_idx = np.random.choice(nonzero_idx, self.n_mask, replace=False)
            for j in mask_idx:
                mask_row_ids.append(i)
                mask_col_ids.append(j)

        assert len(mask_col_ids) == len(mask_row_ids)

        # the genes to predict will be masked out in the input
        gene_targets = gene_targets[mask_row_ids, mask_col_ids].reshape(self.batch_size, -1)

        gene_ids, cell_prop_ids = self._create_gene_cell_prop_ids()

        # target ids are the genes that are masked out plus the cell properties to predict
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

        batch = gene_ids, gene_target_ids, cell_prop_ids, gene_vals, gene_targets, key_padding_mask, cell_prop_vals

        if self.pin_memory:
            for tensor in batch:
                tensor.pin_memory()

        return batch


class AnnDataset(SingleCellDataset):
    def __init__(
        self,
        anndata: Any,
        cell_idx: List[int],
        gene_idx: List[int],
        cells_per_epochs: int,
        predict_classes: Optional[List[str]] = None,
        n_mask: int = 316,
        batch_size: int = 32,
        rank_order: bool = True,
        normalize_total: Optional[float] = 1e4,
        log_normalize: bool = True,
        pin_memory: bool = False,
    ):

        self.anndata = anndata
        self.cell_idx = cell_idx
        self.gene_idx = gene_idx
        self.cells_per_epochs = cells_per_epochs
        self.predict_classes = predict_classes

        self.n_classes = len(predict_classes) if predict_classes is not None else 0
        self.n_mask = n_mask
        self.batch_size = batch_size
        self.rank_order = rank_order
        if rank_order:
            print("Since rank_oder=True, setting normalize_total=None and log_normalize=False")
            normalize_total = None
            log_normalize = False
        self.normalize_total = normalize_total
        self.log_normalize = log_normalize

        self.pin_memory = pin_memory

        self.n_samples = self.anndata.shape[0]
        self.n_genes = len(gene_idx)

        self._get_class_info()
        self._create_gene_class_ids()

    def _get_class_info(self):
        """Extract the list of uniques values for each class (e.g. sex, cell type, etc.) to be predicted"""
        if self.predict_classes is not None:
            self.class_unique = {}
            self.class_dist = {}
            for k in self.predict_classes:
                unique_list, counts = np.unique(self.anndata.obs[k], return_counts=True)
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
                class_vals[n0, n1] = np.where(self.anndata.obs[k][i] == v)[0]

        return torch.from_numpy(class_vals)

    def _get_gene_vals(self, idx: List[int]):

        gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        for n, i in enumerate(idx):
            x = self.anndata[i].X.toarray()
            x = x[:, self.gene_idx]

            if self.rank_order:
                gene_vals[n, :] = self._rank_order(x)
            else:
                gene_vals[n, :] = self._normalize(x)

        zero_idx = np.where(gene_vals == 0)
        gene_vals = torch.from_numpy(gene_vals)
        # return two copies since we'll modify gene_vals but keep gene_targets as is
        return gene_vals, gene_vals, zero_idx

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = x * self.normalize_total / np.sum(x, axis=1, keepdims=True) if self.normalize_total is not None else x
        x = np.log1p(x) if self.log_normalize else x
        return x

    def _rank_order(self, x: np.ndarray) -> np.ndarray:
        """Will assign scores from 0 (lowest) to 1 (highest)."""
        cell_rank = np.zeros_like(x)
        for i in range(x.shape[0]):
            unique_counts = np.unique(x[i, :])
            rank_score = np.linspace(0.0, 1.0, len(unique_counts))
            for n, count in enumerate(unique_counts):
                idx = np.where(x[i, :] == count)[0]
                cell_rank[i, idx] = rank_score[n]

        return cell_rank


class DataModule(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        train_data_path: str,
        train_metadata_path: str,
        test_data_path: str,
        test_metadata_path: str,
        batch_size: int = 32,
        num_workers: int = 16,
        n_mask: int = 100,
        rank_order: bool = False,
        cell_properties: Optional[Dict[str, Any]] = None,

    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.train_metadata_path = train_metadata_path
        self.test_data_path = test_data_path
        self.test_metadata_path = test_metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_mask = n_mask
        self.rank_order = rank_order
        self.cell_properties = cell_properties

    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.train_data_path,
            self.train_metadata_path,
            cell_properties=self.cell_properties,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
        )
        self.val_dataset = SingleCellDataset(
            self.test_data_path,
            self.test_metadata_path,
            cell_properties=self.cell_properties,
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
            RandomSampler(self.val_dataset),
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


class DataModuleAnndata(pl.LightningDataModule):

    # data_path: Path to directory with preprocessed data.
    # classify: Name of column from `obs` table to add classification task with. (optional)
    # Fraction of median genes to mask for prediction.
    # batch_size: Dataloader batch size
    # num_workers: Number of workers for DataLoader.

    def __init__(
        self,
        anndata_path: str,
        batch_size: int = 32,
        num_workers: int = 16,
        n_min_mask: int = 1,
        n_max_mask: int = 100,
        rank_order: bool = False,
        cell_properties: Optional[Dict[str, Any]] = None,
        gene_min_pct_threshold: float = 0.02,
        min_genes_per_cell: int = 1000,
        train_pct: float = 0.9,
    ):
        super().__init__()

        self.anndata = sc.read_h5ad(anndata_path, "r")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_min_mask = n_min_mask
        self.n_max_mask = n_max_mask
        self.rank_order = rank_order
        self.cell_properties = cell_properties
        self.gene_min_pct_threshold = gene_min_pct_threshold
        self.min_genes_per_cell = min_genes_per_cell
        self.train_pct = train_pct

        self._train_test_splits()
        self._get_gene_index()

    def setup(self, stage):

        self.train_dataset = AnnDataset(
            self.anndata,
            self.train_idx,
            self.gene_idx,
            128 * 2000,
            cell_properties=self.cell_properties,
            n_min_mask=self.n_min_mask,
            n_max_mask=self.n_max_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
        )
        self.val_dataset = AnnDataset(
            self.anndata,
            self.test_idx,
            self.gene_idx,
            128 * 100,
            cell_properties=self.cell_properties,
            n_min_mask=self.n_min_mask,
            n_max_mask=self.n_max_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
        )
        self.n_genes = self.train_dataset.n_genes
        print(f"number of genes {self.n_genes}")

    def _train_test_splits(self):

        # TODO: might want to make split by subjects
        n_genes = self.anndata.obs["n_genes"].values
        cell_idx = np.where(n_genes > self.min_genes_per_cell)[0]
        np.random.shuffle(cell_idx)
        n = len(cell_idx)
        self.train_idx = cell_idx[: int(n * self.train_pct)]
        self.test_idx = cell_idx[int(n * self.train_pct):]
        print(f"Number of training cells: {len(self.train_idx)}")
        print(f"Number of test cells: {len(self.test_idx)}")

    def _get_gene_index(self, chunk_size: int = 10_000, n_segments: int = 5):

        n = self.anndata.shape[0]
        start_idx = np.linspace(0, n - chunk_size - 1, n_segments)
        gene_expression = []

        for i in start_idx:
            x = self.anndata[int(i): int(i + chunk_size)].to_memory()
            x = x.X.toarray()
            gene_expression.append(np.mean(x > 0, axis=0))

        gene_expression = np.mean(np.stack(gene_expression), axis=0)
        self.gene_idx = np.where(gene_expression >= self.gene_min_pct_threshold)[0]
        print(f"Number of genes selected: {len(self.gene_idx)}")

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
            RandomSampler(self.val_dataset),
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

