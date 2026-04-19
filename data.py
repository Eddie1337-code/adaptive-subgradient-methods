"""
Data loading and preprocessing for the Avazu CTR dataset.

Features are hashed to fixed-size indices per field using a deterministic
vectorized hash for reproducibility and performance.
Data is streamed in chunks to handle large CSV files efficiently.
"""

import hashlib

import numpy as np
import pandas as pd
import torch

# Complete list of categorical features in the Avazu dataset.
# At runtime, only features actually present in the CSV are used.
_ALL_CATEGORICAL_FEATURES = [
    'C1', 'banner_pos',
    'site_id', 'site_domain', 'site_category',
    'app_id', 'app_category',
    'device_id', 'device_model',
    'device_type', 'device_conn_type',
    'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
]


def _field_seed(field_name: str) -> int:
    """Derive a deterministic per-field seed from the field name."""
    digest = hashlib.md5(field_name.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def _hash_column(values: np.ndarray, field_seed: int,
                 hash_size: int) -> np.ndarray:
    """Vectorized deterministic hash of a string column.

    Strategy: hash only the *unique* values (via MD5, C-implemented and fast),
    then map all rows via a numpy lookup.  This turns O(N) Python hash calls
    into O(U) where U = number of unique values (typically U << N).
    """
    cats = pd.Categorical(values)
    codes = cats.codes.astype(np.int64)
    categories = cats.categories.values

    # Hash each unique category once — MD5 is C-implemented, fast per call
    seed_bytes = field_seed.to_bytes(8, 'little')
    cat_hashes = np.empty(len(categories), dtype=np.int64)
    for i, cat in enumerate(categories):
        digest = hashlib.md5(seed_bytes + str(cat).encode('utf-8')).digest()
        cat_hashes[i] = int.from_bytes(digest[:8], 'little') % hash_size

    # Map codes → hashes (vectorized numpy lookup)
    return cat_hashes[codes]



def detect_features(csv_path: str):
    """Detect which categorical features and labels are present in a CSV.

    Returns:
        features: list of available categorical feature column names
        n_fields: total number of feature fields (categoricals + hour_of_day)
        has_labels: whether the 'click' column is present
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    features = [f for f in _ALL_CATEGORICAL_FEATURES if f in header]
    has_labels = 'click' in header
    n_fields = len(features) + 1  # +1 for hour_of_day
    return features, n_fields, has_labels


def _hash_features(df: pd.DataFrame, features: list,
                   hash_size: int) -> np.ndarray:
    """Hash categorical features and hour into index array.

    Returns:
        indices: np.ndarray of shape (n_samples, n_fields), dtype int64
    """
    n = len(df)
    n_fields = len(features) + 1  # +1 for hour_of_day
    indices = np.zeros((n, n_fields), dtype=np.int64)

    # Precompute per-field seeds once
    field_seeds = {feat: _field_seed(feat) for feat in features}
    field_seeds['hour_of_day'] = _field_seed('hour_of_day')

    for j, feat in enumerate(features):
        col = df[feat].astype(str).values
        indices[:, j] = _hash_column(col, field_seeds[feat], hash_size)

    # hour_of_day
    hour_of_day = (df['hour'].values % 100).astype(str)
    indices[:, -1] = _hash_column(hour_of_day, field_seeds['hour_of_day'],
                                  hash_size)
    return indices


def process_chunk(df: pd.DataFrame, features: list,
                  hash_size: int = 100000) -> tuple:
    """
    Convert a DataFrame chunk to feature index arrays and labels.

    Args:
        df: DataFrame chunk from CSV reader
        features: list of categorical feature column names to use
        hash_size: number of hash buckets per field

    Returns:
        indices: np.ndarray of shape (n_samples, n_fields), dtype int64
        labels:  np.ndarray of shape (n_samples,), dtype float32
    """
    indices = _hash_features(df, features, hash_size)
    labels = df['click'].values.astype(np.float32)
    return indices, labels


def _stratified_sample(df: pd.DataFrame, sample_rate: float,
                       seed: int = 42) -> pd.DataFrame:
    """Stratified sample by (hour, site_domain), preserving original order."""
    parts = []
    for _, group in df.groupby(['hour', 'site_domain']):
        n = max(1, int(len(group) * sample_rate))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sort_index()


def iterate_minibatches(csv_path: str, features: list,
                        batch_size: int = 4096, hash_size: int = 100000,
                        chunk_size: int = 100000, device: str = 'cpu',
                        shuffle_chunks: bool = True, sample_rate: float = 1.0):
    """
    Generator that yields (feature_indices, labels) mini-batches from a CSV.

    Args:
        csv_path: Path to the CSV file
        features: list of categorical feature column names
        batch_size: Mini-batch size
        hash_size: Hash table size per field
        chunk_size: Number of rows to read at a time from CSV
        device: torch device
        shuffle_chunks: Whether to shuffle within each chunk
        sample_rate: Fraction of data to use (1.0 = all, 0.2 = 20%),
                     stratified by (hour, site_domain)

    Yields:
        (indices_tensor, labels_tensor) tuples
    """
    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    for chunk_df in reader:
        if sample_rate < 1.0:
            chunk_df = _stratified_sample(chunk_df, sample_rate)

        indices, labels = process_chunk(chunk_df, features, hash_size)

        if shuffle_chunks:
            perm = np.random.permutation(len(indices))
            indices = indices[perm]
            labels = labels[perm]

        # Transfer entire chunk to device once, then slice (O(1) per batch)
        chunk_idx = torch.from_numpy(indices).to(device)
        chunk_lbl = torch.from_numpy(labels).to(device)
        n = len(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield chunk_idx[start:end], chunk_lbl[start:end]


def iterate_unlabeled(csv_path: str, features: list,
                      batch_size: int = 4096, hash_size: int = 100000,
                      chunk_size: int = 100000, device: str = 'cpu'):
    """
    Generator for unlabeled CSV files (no 'click' column).

    Yields (feature_indices, chunk_df_slice) tuples so that predictions
    can be joined back to the original rows.
    """
    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    for chunk_df in reader:
        indices = _hash_features(chunk_df, features, hash_size)
        chunk_idx = torch.from_numpy(indices).to(device)
        n = len(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield chunk_idx[start:end], chunk_df.iloc[start:end]
