from typing import List, Optional

import os
import pickle
import numpy as np
import scanpy as sc


class CreateData:
    """Class to save the data in numpy memmap format
    Will save the gene expression using int16 to save space,
    but will create the function to normalize the data using
    the gene expression statistics"""

    def __init__(
        self,
        source_fn: str,
        target_path: str,
        train_pct: float = 0.9,
        rank_order: bool = False,
        normalize_total: Optional[float] = 1e4,
        log_normalize: bool = True,
        min_genes_per_cell: int = 1000,
        min_percent_cells_per_gene: float = 2.0,
    ):
        self.source_fn = source_fn
        self.target_path = target_path
        self.train_pct = train_pct
        self.rank_order = rank_order
        if rank_order:
            print("Since rank_oder=True, setting normalize_total=None and log_normalize=False")
            normalize_total = None
            log_normalize = False
        self.normalize_total = normalize_total
        self.log_normalize = log_normalize
        self.min_genes_per_cell = min_genes_per_cell
        self.min_percent_cells_per_gene = min_percent_cells_per_gene

        self.obs_keys = ['CERAD', 'BRAAK_AD', 'Dementia', 'AD', 'class', 'subclass', 'subtype', 'ApoE_gt', 'Sex',
                        'Head_Injury', 'Vascular', 'Age', 'Epilepsy', 'Seizures', 'Tumor']
        # self.var_keys = ['gene_id', 'gene_name', 'gene_type']

        self.anndata = sc.read_h5ad(source_fn, 'r+')

        print(f"Size of anndata {self.anndata.shape[0]}")

    def _train_test_splits(self):
        # TODO: might want to make split by subjects
        n_genes = self.anndata.obs["n_genes"].values
        good_cells = np.where(n_genes > self.min_genes_per_cell)[0]
        np.random.shuffle(good_cells)
        print(f"Original size: {self.anndata.shape[0]}, filtered size: {len(good_cells)}")
        self.train_idx = good_cells[: int(len(good_cells) * self.train_pct)]
        self.test_idx = good_cells[int(len(good_cells) * self.train_pct):]

        self.train_idx = np.sort(self.train_idx)
        self.test_idx = np.sort(self.test_idx)

    def _calculate_stats(self, fp):

        n_cells, n_features = fp.shape

        count = 0
        sum_all = np.zeros(n_features, dtype=np.float32)
        sumsq = np.zeros(n_features, dtype=np.float32)
        self.max_vals = np.zeros(n_features, dtype=np.float32)
        count_nonzero = np.ones(n_features, dtype=np.float32)  # add one to stabilize stats
        for i in range(n_cells):
            y = np.array(fp[i, :]).astype(np.float32)
            count += 1
            sum_all += y
            sumsq += y**2
            idx = np.where(y > 0)[0]
            count_nonzero[idx] += 1
            idx_max = np.where(y > self.max_vals)[0]
            self.max_vals[idx_max] = y[idx_max]

        self.mean = sum_all / count
        self.mean_nonzero = sum_all / count_nonzero
        self.std = np.sqrt((sumsq / count) - (self.mean**2))
        self.std_nonzero = np.sqrt((sumsq / count_nonzero) - (self.mean_nonzero ** 2))

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

    def _create_metadata(self, train: bool = True):

        idx = self.train_idx if train else self.test_idx
        idx_genes = self._filter_genes()
        meatadata_fn = "train_metadata.pkl" if train else "test_metadata.pkl"
        meatadata_fn = os.path.join(self.target_path, meatadata_fn)

        meta = {
            "obs": {k: self.anndata.obs[k][idx].values for k in self.obs_keys},
            # "var": {k: self.anndata.var[k][idx_genes].values for k in self.var_keys},
            "var": self.anndata.var,
            "stats": {
                "mean": self.mean,
                "std": self.std,
                "max": self.max_vals,
                "mean_nonzero": self.mean,
                "std_nonzero": self.std,
                "normalize_total": self.normalize_total,
                "log_normalize": self.log_normalize,
                "rank_order": self.rank_order,
            },
        }
        k = self.obs_keys[0]
        pickle.dump(meta, open(meatadata_fn, "wb"))

    def _filter_genes(self):
        p = self.anndata.var["percent_cells"].values
        return np.where(p >= self.min_percent_cells_per_gene)[0]

    def _create_dataset(self, train: bool = True):

        idx = self.train_idx if train else self.test_idx
        data_fn = "train_data.dat" if train else "test_data.dat"
        data_fn = os.path.join(self.target_path, data_fn)
        # idx_genes = self._filter_genes()
        idx_genes = np.arange(self.anndata.shape[1])

        chunk_size = 25_000  # chunk size for loading data into memory
        fp = np.memmap(data_fn, dtype='float16', mode='w+', shape=(len(idx), len(idx_genes)))

        for n in range(len(idx) // chunk_size + 1):
            m = np.minimum(len(idx), (n + 1) * chunk_size)
            current_idx = idx[n * chunk_size: m]
            y = self.anndata[current_idx].to_memory()
            y = y.X.toarray().astype(np.float32)
            y = y[:, idx_genes]
            if self.rank_order:
                y = self._rank_order(y)
            else:
                y = self._normalize(y)
            fp[n * chunk_size: m, :] = y.astype(np.float16)

        # flush to memory
        fp.flush()

        return fp

    def create_datasets(self) -> None:

        # randomly choose the train/test splits
        self._train_test_splits()

        print("Saving the training data in the memmap array...")
        # must create the training set first, since gene stats are calculated using
        fp = self._create_dataset(train=True)

        print("Calculating the gene expression stats...")
        self._calculate_stats(fp)

        print("Saving the training metadata...")
        self._create_metadata(train=True)

        print("Saving the test data in the memmap array...")
        _ = self._create_dataset(train=False)

        print("Saving the test metadata...")
        self._create_metadata(train=False)


if __name__ == "__main__":

    source_path = "/sc/arion/projects/psychAD/NPS-AD/freeze2_rc/h5ad_final/FULL_2023-06-08_23_53_original.h5ad"
    target_path = "/sc/arion/projects/psychAD/massen06/perceiver_data"
    c = CreateData(source_path, target_path, train_pct=0.95, rank_order=True)

    c.create_datasets()


