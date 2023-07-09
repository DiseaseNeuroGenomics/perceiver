from typing import Any, Dict, List, Optional, Tuple, Union

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
    Subset,
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
        pin_memory: bool = False,
        cell_prop_same_ids: bool = False,
        max_cell_prop_val: float = 999,
        cutmix_pct: float = 0.0,
        mixup: bool = False,
        max_gene_val: float = 10.0,
        protein_coding_only: bool = True,
        bin_gene_count: bool = True,
        n_gene_bins: int = 16,
    ):

        self.metadata = pickle.load(open(metadata_path, "rb"))
        self.data_path = data_path
        self.cell_properties = cell_properties
        self.n_samples = len(self.metadata["obs"]["class"])
        print(f"Number of cells {self.n_samples}")
        if "gene_name" in self.metadata["var"].keys():
            self.n_genes_original = len(self.metadata["var"]["gene_name"])
        else:
            self.n_genes_original = len(self.metadata["var"])
        self.n_cell_properties = len(cell_properties) if cell_properties is not None else 0
        self.n_mask = n_mask
        self.batch_size = batch_size

        if bin_gene_count:
            print("Since bin_gene_count is True, setting rank_oder to False")
            rank_order = False
        if rank_order:
            print("Since rank_oder is True, setting all other normalization to False")
            normalize_total = None
            log_normalize = False
            bin_gene_count = False
        if normalize_total or log_normalize:
            print("Since log_normalize or normalize_total is True, setting all other normalization to False")
            rank_order = False

        self.normalize_total = normalize_total
        self.log_normalize = log_normalize
        self.rank_order = rank_order
        self.bin_gene_count = bin_gene_count
        self.n_gene_bins = n_gene_bins
        self.pin_memory = pin_memory
        self.cell_prop_same_ids = cell_prop_same_ids
        self.max_cell_prop_val = max_cell_prop_val
        self.cutmix_pct = cutmix_pct
        self.mixup = mixup
        self.max_gene_val = max_gene_val
        self.protein_coding_only = protein_coding_only

        self.offset = 2 * self.n_genes_original  # FP16 is 2 bytes

        # possibly use for embedding the gene inputs
        self.cell_classes = np.array(['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo'])

        # this will down-sample the number if genes if specified
        # for now, need to call this AFTER calculating offset
        self._get_gene_index()

        self._get_cell_prop_info()
        self._create_gene_cell_prop_ids()


        self.bins = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 13, 16, 22, 35, 60, 9999]

    def __len__(self):
        return self.n_samples

    def _get_gene_index(self):

        if self.protein_coding_only:
            self.gene_idx = np.where(self.metadata["var"]['protein_coding'])[0]
        else:
            self.gene_idx = np.arange(self.n_genes_original)
        self.n_genes = len(self.gene_idx)
        print(f"Sub-sampling genes. Number of genes is now {self.n_genes}")

    def _create_gene_cell_prop_ids(self):
        """"Create the gene and class ids. Will start with the gene ids, and then concatenate
        the cell property ids if requested"""
        gene_ids = torch.arange(0, self.n_genes).repeat(self.batch_size, 1)
        if self.n_cell_properties > 0:
            if self.cell_prop_same_ids:
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
                # skip if required field are already present as this function can be called multiple
                # times if using multiple GPUs
                if "freq" in self.cell_properties[k] or "mean" in self.cell_properties[k]:
                    continue
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
                            isinstance(u, str) or (u >= 0 and u < self.max_cell_prop_val)
                        )
                    ]
                    self.cell_properties[k]["values"] = unique_list[idx]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print("CELL PROP INFO",k, self.cell_properties[k]["freq"])

                elif cell_prop["discrete"] and cell_prop["values"] is not None:
                    unique_list, counts = np.unique(self.metadata["obs"][k], return_counts=True)
                    idx = [n for n, u in enumerate(unique_list) if u in cell_prop["values"]]
                    self.cell_properties[k]["freq"] = counts[idx] / np.mean(counts[idx])
                    print("CELL PROP INFO", k, self.cell_properties[k]["freq"])

        else:
            self.cell_prop_dist = None

    def _get_cell_prop_vals(self, batch_idx: List[int]):
        """Extract the cell property value for ach entry in the batch"""
        if self.n_cell_properties == 0:
            return None

        p_dims = [len(p["values"]) for p in self.cell_properties.values()]

        cell_prop_vals = np.zeros((self.batch_size, self.n_cell_properties, np.max(p_dims)), dtype=np.float32)
        for n0, i in enumerate(batch_idx):
            for n1, (k, cell_prop) in enumerate(self.cell_properties.items()):
                if not cell_prop["discrete"]:
                    # continuous value
                    cell_val = self.metadata["obs"][k][i]
                    if np.abs(cell_val) > self.max_cell_prop_val:
                        cell_prop_vals[n0, n1, 0] = -9999
                    else:
                        # normalize
                        cell_prop_vals[n0, n1, 0] = (cell_val - cell_prop["mean"]) / cell_prop["std"]
                else:
                    # discrete value
                    idx = np.where(self.metadata["obs"][k][i] == np.array(cell_prop["values"]))[0]
                    # cell property values of -1 will imply N/A, and will be masked out
                    if len(idx) == 0:
                        cell_prop_vals[n0, n1, 0] = -9999
                    else:
                        cell_prop_vals[n0, n1, idx[0]] = 1.0


        cell_class_id = np.zeros((self.batch_size,), dtype=np.int64)
        for n, i in enumerate(batch_idx):
            idx = np.where(self.metadata["obs"]["class"][i] == self.cell_classes)[0]
            cell_class_id[n] = idx[0]

        return torch.from_numpy(cell_prop_vals), torch.from_numpy(cell_class_id)

    def _get_gene_vals(self, batch_idx: List[int]):

        gene_vals = np.zeros((self.batch_size, self.n_genes), dtype=np.float32)
        input_gene_vals = np.zeros_like(gene_vals)

        x = np.memmap(
            self.data_path, dtype='float16', mode='r', shape=(1000, self.n_genes_original,), offset=0 * self.offset
        )[:, self.gene_idx].astype(np.float32)


        for n, i in enumerate(batch_idx):
            gene_vals[n, :] = np.memmap(
                self.data_path, dtype='float16', mode='r', shape=(self.n_genes_original,), offset=i * self.offset
            )[self.gene_idx].astype(np.float32)
            if self.bin_gene_count:
                input_gene_vals[n, :] = self._bin_gene_count(gene_vals[n, :])
                gene_vals[n, :] = self._normalize(gene_vals[n, :])
            elif self.rank_order:
                input_gene_vals[n, :] = self._rank_order(gene_vals[n, :])
                gene_vals[n, :] = self._normalize(gene_vals[n, :])
            elif self.normalize_total or self.log_normalize:
                input_gene_vals[n, :] = self._normalize(gene_vals[n, :])
                gene_vals[n, :] = input_gene_vals[n, :]


        # return two copies since we'll modify gene_vals but keep gene_targets as is
        return torch.from_numpy(input_gene_vals), torch.from_numpy(gene_vals)

    def _bin_gene_count(self, x: np.ndarray) -> np.ndarray:
        return np.digitize(x, self.bins)


    def _rank_order(self, x: np.ndarray) -> np.ndarray:
        """Expression values of 0 are mapped to 0. Expression values > 0 will be mapped to the percentage of
        non-zero genes they're greater than"""
        cell_rank = np.zeros_like(x)
        vals, counts = np.unique(x, return_counts=True)
        counts = counts[vals > 0]
        vals = vals[vals > 0]
        total_sum = np.sum(counts)
        for val, count in zip(vals, counts):
            s = np.sum(counts[val > vals]) / total_sum
            idx = np.where(x == val)[0]
            cell_rank[idx] = np.clip(s, 0.1, 1.0) ** 2

        return cell_rank

    def _normalize(self, x: np.ndarray) -> np.ndarray:

        x = x * self.normalize_total / np.sum(x) if self.normalize_total is not None else x
        x = np.log1p(x) if self.log_normalize else x
        x = np.minimum(x, self.max_gene_val)

        return x

    def _prepare_data(self, batch_idx):

        gene_vals, gene_targets = self._get_gene_vals(batch_idx)
        cell_prop_vals, cell_class_id = self._get_cell_prop_vals(batch_idx)

        # If specified, perform data augmentation mixup or cutmix
        if self.mixup:
            gene_vals, gene_targets, cell_prop_vals = self._mixup(gene_vals, gene_targets, cell_prop_vals)
        elif self.cutmix_pct > 0:
            gene_vals, gene_targets, cell_prop_vals = self._cutmix(gene_vals, gene_targets, cell_prop_vals)

        zero_idx = np.where(gene_vals == 0)
        key_padding_mask = torch.zeros_like(gene_vals).detach()
        key_padding_mask[zero_idx[0], zero_idx[1]] = 1.0

        return gene_vals, key_padding_mask, gene_targets, cell_prop_vals, cell_class_id

    def _mixup(self, gene_vals, gene_targets, cell_prop_vals):

        # determine the cells with no missing values
        p = cell_prop_vals[:, :, 0].detach().numpy()
        good_idx = list(np.where(np.sum(p < -999, axis=1) == 0)[0])
        good_set = set(good_idx)

        new_cell_prop_vals = torch.zeros_like(cell_prop_vals)
        new_gene_targets = torch.zeros_like(gene_targets)
        new_gene_vals = torch.zeros_like(gene_vals)

        for n in range(self.batch_size):
            if n in good_idx:
                # randomly choose partner
                j = np.random.choice(list(good_set.difference(set([n]))))
                # set mix percentage
                alpha = np.clip(np.random.exponential(0.05), 0.0, 0.25)
                new_gene_vals[n, :] = (1 - alpha) * gene_vals[n, :] + alpha * gene_vals[j, :]

                # take the weighted average of the targets
                new_cell_prop_vals[n, :] = (1 - alpha) * cell_prop_vals[n, :] + alpha * cell_prop_vals[j, :]
                new_gene_targets[n, :] = (1 - alpha) * gene_targets[n, :] + alpha * gene_targets[j, :]

            else:
                new_cell_prop_vals[n, :] = cell_prop_vals[n, :]
                new_gene_targets[n, :] = gene_targets[n, :]
                new_gene_vals[n, :] = gene_vals[n, :]

        return new_gene_vals, new_gene_targets, new_cell_prop_vals

    def _cutmix(self, gene_vals, gene_targets, cell_prop_vals, continuous_block: bool = False):

        # determine the cells with no missing values
        p = cell_prop_vals[:, :, 0].detach().numpy()
        good_idx = list(np.where(np.sum(p < -999, axis=1) == 0)[0])
        good_set = set(good_idx)

        new_cell_prop_vals = torch.zeros_like(cell_prop_vals)
        new_gene_targets = torch.zeros_like(gene_targets)
        new_gene_vals = torch.zeros_like(gene_vals)

        for n in range(self.batch_size):
            if n in good_idx and np.random.rand() < self.cutmix_pct:
                # randomly choose partner
                j = np.random.choice(list(good_set.difference(set([n]))))
                # set mix percentage
                alpha = np.random.uniform(0.005, 0.995)

                # mix-up gene values
                new_gene_vals[n, :] = gene_vals[n, :]
                if continuous_block:
                    start_idx = np.random.randint(0, int(alpha * self.n_genes) - 1)
                    end_idx = start_idx + int((1 - alpha) * self.n_genes)
                    new_gene_vals[n, :] = gene_vals[n, :]
                    new_gene_vals[n, start_idx : end_idx] = gene_vals[j, start_idx: end_idx]
                else:
                    idx_mix = np.random.choice(self.n_genes, int(alpha * self.n_genes), replace=False)
                    new_gene_vals[n, idx_mix] = gene_vals[j, idx_mix]

                # ensure it has enough non-zero entries if not; then revert
                if torch.sum(new_gene_vals[n, :] > 0) < self.n_mask:
                    new_gene_vals[n, :] = gene_vals[n, :]
                    continue

                # take the weighted average of the targets
                new_cell_prop_vals[n, :] = (1 - alpha) * cell_prop_vals[n, :] + alpha * cell_prop_vals[j, :]
                new_gene_targets[n, :] = (1 - alpha) * gene_targets[n, :] + alpha * gene_targets[j, :]


            else:
                new_cell_prop_vals[n, :] = cell_prop_vals[n, :]
                new_gene_targets[n, :] = gene_targets[n, :]
                new_gene_vals[n, :] = gene_vals[n, :]

        return new_gene_vals, new_gene_targets, new_cell_prop_vals

    def __getitem__(self, idx: Union[int, List[int]]):

        if isinstance(idx, int):
            idx = [idx]

        if len(idx) != self.batch_size:
            raise ValueError("Index length not equal to batch_size")

        gene_vals, key_padding_mask, gene_targets, cell_prop_vals, cell_class_id = self._prepare_data(idx)

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


        n_input_genes = np.random.choice([600, 700, 800, 900])
        keep_col_ids = []
        keep_row_ids = []
        for i in range(self.batch_size):
            nonzero_idx = torch.where(gene_vals[i, :] > 0)[0]
            # randomly choose number to mask for each cell
            # n = np.random.randint(0, self.n_mask)
            idx = np.random.choice(nonzero_idx, n_input_genes, replace=False)
            for j in idx:
                keep_row_ids.append(i)
                keep_col_ids.append(j)

        gene_ids = gene_ids[keep_row_ids, keep_col_ids].reshape(self.batch_size, -1)
        gene_vals = gene_vals[keep_row_ids, keep_col_ids].reshape(self.batch_size, -1)
        key_padding_mask = key_padding_mask[keep_row_ids, keep_col_ids].reshape(self.batch_size, -1)

        batch = (
            gene_ids,
            gene_target_ids,
            cell_prop_ids,
            gene_vals,
            gene_targets,
            key_padding_mask,
            cell_prop_vals,
            cell_class_id,
        )

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
        cell_prop_same_ids: bool = False,
        cutmix_pct: float = 0.0,
        mixup: bool = False,
        bin_gene_count: bool = False,
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
        self.cell_prop_same_ids = cell_prop_same_ids
        self.cutmix_pct = cutmix_pct
        self.mixup = mixup
        self.bin_gene_count = bin_gene_count


    def setup(self, stage):

        self.train_dataset = SingleCellDataset(
            self.train_data_path,
            self.train_metadata_path,
            cell_properties=self.cell_properties,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
            cell_prop_same_ids=self.cell_prop_same_ids,
            cutmix_pct=self.cutmix_pct,
            mixup=self.mixup,
            bin_gene_count=self.bin_gene_count,
        )
        self.val_dataset = SingleCellDataset(
            self.test_data_path,
            self.test_metadata_path,
            cell_properties=self.cell_properties,
            n_mask=self.n_mask,
            batch_size=self.batch_size,
            rank_order=self.rank_order,
            pin_memory=False,
            cell_prop_same_ids=self.cell_prop_same_ids,
            cutmix_pct=0.0,
            mixup=False,
            bin_gene_count=self.bin_gene_count,
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
        same_latent_class: bool = False,
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
        self.same_latent_class = same_latent_class

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
            same_latent_class=same_latent_class,
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
            same_latent_class=same_latent_class,
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

