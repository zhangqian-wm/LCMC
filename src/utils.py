import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def projsplx(y):
    """
    projsplx projects a vector to a simplex
    by the algorithm presented in (Chen an Ye, "Projection Onto A Simplex", 2011)
    Author: Sangwoong Yoon (sangwoong24yoon@gmail.com)
    """
    assert len(y.shape) == 1
    N = y.shape[0]
    y_flipsort = np.flipud(np.sort(y))
    cumsum = np.cumsum(y_flipsort)
    t = (cumsum - 1) / np.arange(1, N + 1).astype('float')
    t_iter = t[:-1]
    t_last = t[-1]
    y_iter = y_flipsort[1:]
    if np.all((t_iter - y_iter) < 0):
        t_hat = t_last
    else:
        # find i such that t>=y
        eq_idx = np.searchsorted(t_iter - y_iter, 0, side='left')
        t_hat = t_iter[eq_idx]
    x = y - t_hat
    # there may be a numerical error such that the constraints are not exactly met.
    x[x < 0.] = 0.
    x[x > 1.] = 1.
    assert np.abs(x.sum() - 1.) <= 1e-5
    assert np.all(x >= 0) and np.all(x <= 1.)
    return x


def proj_vh(vh):
    """
    Projection matrix V into the feasible set \mathcal{F}_V

    :param vh: matrix with dimension: (number of quasispecies) x (4*haplotype length)
    :return: matrix in the same dimension of input

    *If all the element are with the same value, the function will let the first element to 1 and others 0
    """
    (k, m) = vh.shape
    vh = np.reshape(vh, (k, -1, 4))
    # vh = np.reshape(vh, (k, int(m / 4), 4))
    # Find the indices of the maximum value along the third dimension
    max_indices = np.argmax(vh, axis=2)

    # Create a new array B with the same shape as A, initialized with zeros
    ind_mat = np.zeros_like(vh)

    # Set the value to 1 where the maximum value occurs
    ind_mat[np.arange(vh.shape[0])[:, None], np.arange(vh.shape[1]), max_indices] = 1
    vh = np.reshape(ind_mat, (k, m))
    return vh


def mat2ten(M):
    T_M = np.dstack((np.double(M == 1), np.double(M == 2), np.double(M == 3), np.double(M == 4))).reshape(M.shape[0], -1)
    return T_M


def ten2mat(T_M):
    M = np.argmax(T_M.reshape(T_M.shape[0], -1, 4), axis=2) + 1
    return M

def _random_partition(y, n_workers=5, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(y.shape[0])          # 打乱样本顺序
    idx_splits = np.array_split(perm, n_workers) # 均分索引
    return idx_splits

def _dirichlet_noniid_partition(
    y: np.ndarray,
    num_clients: int = 10,
    alpha: float = 0.5,
    min_size: int = 10,
    seed: int = 42,
    max_retries: int = 2000,
):
    """
    Dirichlet non-IID partition with a minimum sample size constraint per client.

    Args:
        y: labels, shape (N,), integer class ids
        num_clients: number of clients/partitions
        alpha: Dirichlet concentration parameter (smaller -> more non-IID)
        min_size: minimum number of samples per client
        seed: random seed
        max_retries: max attempts to satisfy min_size

    Returns:
        client_indices: list of np.ndarray indices for each client
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    N = len(y)

    classes = np.unique(y)
    K = len(classes)

    if min_size * num_clients > N:
        raise ValueError(f"min_size*num_clients={min_size*num_clients} exceeds N={N}.")

    # Pre-collect indices per class for speed
    class_to_indices = {c: np.where(y == c)[0] for c in classes}

    for _ in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            idx = class_to_indices[c].copy()
            rng.shuffle(idx)

            # Dirichlet proportions for this class
            props = rng.dirichlet(alpha * np.ones(num_clients))

            # Convert proportions -> split points
            split_points = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            splits = np.split(idx, split_points)

            for j in range(num_clients):
                client_indices[j].extend(splits[j].tolist())

        sizes = np.array([len(ci) for ci in client_indices])
        if sizes.min() >= min_size:
            return [np.array(ci, dtype=int) for ci in client_indices]

    raise RuntimeError(
        f"Failed to satisfy min_size={min_size} after {max_retries} retries. "
        f"Try smaller min_size, larger alpha, or fewer clients."
    )



def _shard_noniid_partition(
    y: np.ndarray,
    num_clients: int = 10,
    shards_per_client: int = 2,
    seed: int = 42,
    shuffle_shards: bool = True,
):
    """
    Shard-based non-IID partition without dropping any samples.
    Uses np.array_split to split into nearly equal shards covering all N samples.

    Args:
        y: labels, shape (N,), integer class ids
        num_clients: number of clients/partitions
        shards_per_client: number of shards per client (smaller -> more non-IID)
        seed: random seed
        shuffle_shards: whether to shuffle shard assignment

    Returns:
        client_indices: list of np.ndarray indices for each client
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    N = len(y)

    num_shards = num_clients * shards_per_client
    if num_shards > N:
        raise ValueError(f"num_shards={num_shards} exceeds N={N}; reduce shards_per_client or clients.")

    # Sort indices by label (classic shard trick)
    idx_sorted = np.argsort(y)

    # Split into shards covering all samples (no drop). Sizes differ by at most 1.
    shards = np.array_split(idx_sorted, num_shards)

    shard_ids = np.arange(num_shards)
    if shuffle_shards:
        rng.shuffle(shard_ids)

    client_indices = []
    for i in range(num_clients):
        take = shard_ids[i * shards_per_client:(i + 1) * shards_per_client]
        client_idx = np.concatenate([shards[s] for s in take], axis=0)
        client_indices.append(client_idx.astype(int))

    # Sanity check: cover all samples exactly once
    all_idx = np.concatenate(client_indices)
    if len(all_idx) != N or len(np.unique(all_idx)) != N:
        raise RuntimeError("Shard partition failed to cover all samples exactly once.")

    return client_indices

def print_label_stats(y, clients, name):
    print(f"\n=== {name} ===")
    for i, idx in enumerate(clients):
        uniq, cnt = np.unique(y[idx], return_counts=True)
        print(f"Client {i:02d}: n={len(idx)}, labels={len(uniq)}, dist={dict(zip(uniq.tolist(), cnt.tolist()))}")


def _stratified_partition(
    y: np.ndarray,
    num_clients: int = 10,
    seed: int = 42,
    shuffle: bool = True
):
    """
    Stratified partition for classification dataset.
    Each client gets approximately the same label distribution as the global dataset.

    Args:
        y (np.ndarray): labels (N,), integer class ids
        num_clients (int): number of partitions
        seed (int): random seed
        shuffle (bool): whether to shuffle samples within each class

    Returns:
        client_indices (list[np.ndarray]): indices for each client
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        class_indices = np.where(y == c)[0]

        if shuffle:
            rng.shuffle(class_indices)

        # Split this class evenly across clients
        splits = np.array_split(class_indices, num_clients)

        for client_id in range(num_clients):
            client_indices[client_id].extend(splits[client_id])

    # Convert to numpy arrays
    client_indices = [np.array(idx, dtype=int) for idx in client_indices]

    return client_indices

def split_rows_for_workers(X, y, n_workers=5, seed=0, partition='random', alpha=0.3, min_size=50, shards_per_client=2):
    if partition == 'random':
        idx_splits = _random_partition(y, n_workers=n_workers, seed=seed)
    # Partition based on label
    elif partition == 'dirichlet': # Dirichlet (min size)
        idx_splits = _dirichlet_noniid_partition(y, num_clients=n_workers, alpha=alpha, min_size=min_size, seed=seed)
    elif partition == 'shard': # Shard-based (no drop)
        idx_splits = _shard_noniid_partition(y, num_clients=n_workers, shards_per_client=shards_per_client, seed=seed)
    elif partition == 'stratified':
        idx_splits = _stratified_partition(y, num_clients=n_workers, seed=seed)
    else:
        raise RuntimeError('Partition not recognized')
    X_parts = [X[idx, :] for idx in idx_splits]  # 每份 (≈1990, 29909)
    return X_parts, idx_splits
    # for i, Xi in enumerate(X_parts):
    #     print(f"worker {i}: {Xi.shape}")


## Preprocessing
def find_identical_nonzero_columns(X, missing_value=0):
    """
    X: (n, m), missing 用 0 表示
    返回:
      ident_cols: (m,) bool，True 表示该列在非零位置上值都相同
      ident_vals: (m,) float/同dtype，每列的那个“相同的非零值”（若该列全是missing则为missing_value）
      nnz_count: (m,) 每列非零个数
    """
    X = np.asarray(X)
    mask = (X != missing_value)          # 非missing
    nnz_count = mask.sum(axis=0)         # 每列观测数

    # 每列第一个非零的位置（如果全零，这里会是 0）
    first_pos = np.argmax(mask, axis=0)

    # 每列的“代表值”：第一个非零值（全零列这里会取到 X[0, j]，但会被 nnz_count==0 处理掉）
    rep = X[first_pos, np.arange(X.shape[1])]

    # 检查：对每列，在非零位置上是否都等于 rep
    # 对 missing 位置不做要求
    ok = (~mask) | (X == rep)            # (n, m)
    ident_cols = ok.all(axis=0)          # (m,)

    # 如果某列 nnz_count==0（全 missing），也可视为“identical”（不会影响计算）
    # ident_cols 已经会是 True（因为 (~mask) 全 True），这里仅保证值合理：
    ident_vals = rep.copy()
    ident_vals[nnz_count == 0] = missing_value
    return ident_cols, ident_vals, nnz_count


def strip_identical_columns(X, ident_cols):
    X = np.asarray(X)
    keep = ~ident_cols
    X_reduced = X[:, keep]
    return X_reduced, keep


def extract_low_coverage_columns(
    X,
    threshold=0.30,
    missing_value=0,
    return_stats=False,
    return_mask=True
):
    """
    Extract column indices with coverage ratio lower than threshold.

    Coverage ratio of column j:
        (# rows where X[:, j] != missing_value) / n_rows

    Parameters
    ----------
    X : array-like, shape (n_rows, n_cols)
    threshold : float
        Coverage ratio threshold in [0, 1]. Columns with ratio < threshold are selected.
    missing_value : scalar
        Value used for missing entries (default 0).
    return_stats : bool
        If True, also return coverage_ratio and coverage_count for selected columns.
    return_mask : bool
        If True, return a boolean mask with length n_cols.
        If False, return index array.

    Returns
    -------
    idx_or_mask : np.ndarray
        Boolean mask (if return_mask=True) or index array (if return_mask=False).
    coverage_ratio_selected : np.ndarray, optional
    coverage_count_selected : np.ndarray, optional
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1]")

    n_rows = X.shape[0]
    coverage_count = np.sum(X != missing_value, axis=0)
    coverage_ratio = coverage_count / n_rows
    idx = np.where(coverage_ratio < threshold)[0]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[idx] = True
    out = mask if return_mask else idx

    if return_stats:
        return out, coverage_ratio[idx], coverage_count[idx]
    return out


def restore_identical_columns(Z_reduced, keep, ident_cols, ident_vals, X_original=None, missing_value=0):
    """
    Z_reduced: (n, sum(keep)) 对保留列计算后的结果
    keep: (m,) bool，True 表示保留列
    ident_cols: (m,) bool
    ident_vals: (m,) 每个 identical 列的常量非零值
    X_original: 如果你希望恢复时严格保留原始的missing(0)模式，传入原X即可
    """
    n = Z_reduced.shape[0]
    m = keep.size
    Z_full = np.zeros((n, m), dtype=Z_reduced.dtype)

    # 放回计算过的列
    Z_full[:, keep] = Z_reduced

    if X_original is None:
        # 不提供原X：identical列直接整列填常量（注意这会把原missing也填掉）
        Z_full[:, ident_cols] = ident_vals[ident_cols]
    else:
        # 提供原X：只在原来非missing处填常量，missing处仍为0
        mask = (X_original != missing_value)
        cols = np.where(ident_cols)[0]
        for j in cols:
            Z_full[mask[:, j], j] = ident_vals[j]
            # missing 位置保持 0

    return Z_full


def restore_centers(C_reduced, keep, ident_cols, ident_vals):
    """
    C_reduced: (5, p)
    keep: (m,) bool, True=保留列
    ident_cols: (m,) bool, True=identical列（被剔除）
    ident_vals: (m,) 记录的 identical 列常量值
    return: C_full (5, m)
    """
    k, p = C_reduced.shape
    m = keep.size
    assert p == keep.sum(), "C_reduced 的列数 p 必须等于 keep.sum()"

    C_full = np.zeros((k, m), dtype=C_reduced.dtype)

    # 1) 把 reduced 的中心放回保留列
    C_full[:, keep] = C_reduced

    # 2) identical 列填回常量值
    C_full[:, ident_cols] = ident_vals[ident_cols]  # 广播到 5 行

    return C_full


def assign_and_count(M, V, chunk_size=None):
    """
    Assign each row of M to the argmax match-count cluster in V, and count frequencies.

    M: (n, m) int array
    V: (k, m) int array
    chunk_size: None (fast, but may use large memory) or int (memory-safe)

    Returns
    -------
    labels: (n,) int, in {0,...,k-1}
    freq:   (k,) int, counts of each label
    """
    M = np.asarray(M)
    V = np.asarray(V)
    n, m = M.shape
    k = V.shape[0]
    assert V.shape[1] == m, "V must have the same number of columns as M"

    # Memory-safe path: compute labels in chunks to avoid (n,k,m) boolean tensor
    if chunk_size is not None:
        labels = np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            U = (M[s:e, None, :] == V[None, :, :]).sum(axis=2)  # (chunk, k)
            labels[s:e] = np.argmax(U, axis=1)
    else:
        # Fast vectorized path (may allocate (n,k,m) bool)
        U = (M[:, None, :] == V[None, :, :]).sum(axis=2)        # (n, k)
        labels = np.argmax(U, axis=1)

    # max_vals = U.max(axis=1, keepdims=True)
    # is_max = (U == max_vals)
    # labels = np.array([ np.random.choice(np.flatnonzero(row)) for row in is_max ])

    freq = np.bincount(labels, minlength=k)
    return labels, freq


def compute_MEC(M, V, P=None, label=None):
    V = np.atleast_2d(V)
    assert V.shape[1] == M.shape[1], "Dimensions of V and M are not match"
    if V.shape[0] == 1:
        if P is not None:
            return np.sum((M != V[0])[P])
        return np.sum(M != V[0])

    if label is None:
        label,_ = assign_and_count(M, V)
    if P is not None:
        return np.sum((V[label, :] != M)[P])

    return np.sum(V[label] != M)

def compute_SSE(M, V, P=None, label=None):
    P = None
    M = np.asarray(M)
    V = np.atleast_2d(np.asarray(V))
    assert V.shape[1] == M.shape[1]
    if P is not None:
        P = np.asarray(P, dtype=bool)
        assert P.shape == M.shape
    # 用 dtype 推断更稳：若 V 是 float，结果用 float；若都是 int，至少用 float32 防溢出
    dtype = np.result_type(M.dtype, V.dtype, np.float32)
    work = np.empty_like(M, dtype=dtype)
    if V.shape[0] == 1:
        np.subtract(M, V[0], out=work)
    else:
        if label is None:
            label, _ = assign_and_count(M, V)
        np.subtract(M, V[label], out=work)
    if P is not None:
        return np.sum((work * work)[P])
    return np.sum(work * work)


def sorted_svds(A, k, rng=None):
    # if rng is None:
    #     rng = np.random.default_rng()
    # v0 = rng.standard_normal(min(A.shape))
    # U, S, Vt = svds(A, k=k, which='LM', v0=v0)
    U, S, Vt = svds(A, k=k, which='LM')
    idx = np.argsort(S)[::-1]
    return U[:, idx], S[idx], Vt[idx]






### Performance Evaluation
def centroid_multiclass_mode(X, P, missing_val=0):
    """
    X: (r, m) int in {0,1,2,3,4}
    P: (r, m) bool, True=observed
    return:
        centroid: (m,) int in {1,2,3,4} or missing_val if no observation
    """
    r, m = X.shape
    centroid = np.full(m, missing_val, dtype=X.dtype)

    # 统计每个类别在每列的出现次数
    counts = np.zeros((4, m), dtype=np.int32)  # 对应类别 1,2,3,4

    for v in range(1, 5):
        counts[v-1] = np.sum((X == v) & P, axis=0)

    # 找到每列出现最多的类别
    max_counts = counts.max(axis=0)

    # 有观测的列
    observed_cols = max_counts > 0

    # argmax 得到类别索引 0..3 → 对应 1..4
    centroid[observed_cols] = np.argmax(counts[:, observed_cols], axis=0) + 1

    return centroid


def _contingency_matrix(true_label, pred_label):
    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    if true_label.shape != pred_label.shape:
        raise ValueError("true_label and pred_label must have the same shape")
    if true_label.ndim != 1:
        true_label = true_label.ravel()
        pred_label = pred_label.ravel()

    true_classes, true_inv = np.unique(true_label, return_inverse=True)
    pred_classes, pred_inv = np.unique(pred_label, return_inverse=True)
    cm = np.zeros((true_classes.size, pred_classes.size), dtype=np.int64)
    np.add.at(cm, (true_inv, pred_inv), 1)
    return cm, true_classes, pred_classes


def align_labels_hungarian(true_label, pred_label):
    """
    Align predicted cluster IDs to true IDs using Hungarian matching.

    Returns:
        pred_aligned: aligned predicted labels
        mapping: dict {pred_class: true_class}
    """
    true_label = np.asarray(true_label).ravel()
    pred_label = np.asarray(pred_label).ravel()
    cm, true_classes, pred_classes = _contingency_matrix(true_label, pred_label)

    n = max(cm.shape) if cm.size else 0
    if n == 0:
        return pred_label.copy(), {}

    score = np.zeros((n, n), dtype=np.int64)
    score[:cm.shape[0], :cm.shape[1]] = cm
    row_ind, col_ind = linear_sum_assignment(-score)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < true_classes.size and c < pred_classes.size:
            mapping[pred_classes[c]] = true_classes[r]

    pred_aligned = np.array([mapping.get(x, x) for x in pred_label], dtype=pred_label.dtype)
    return pred_aligned, mapping


def _comb2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def _adjusted_rand_index(true_label, pred_label):
    cm, _, _ = _contingency_matrix(true_label, pred_label)
    n = cm.sum()
    if n <= 1:
        return 1.0

    nij = cm.astype(np.float64)
    ai = nij.sum(axis=1)
    bj = nij.sum(axis=0)

    sum_nij = _comb2(nij).sum()
    sum_ai = _comb2(ai).sum()
    sum_bj = _comb2(bj).sum()
    total = _comb2(n)
    if total == 0:
        return 1.0

    expected = (sum_ai * sum_bj) / total
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_nij - expected) / denom)


def _normalized_mutual_info(true_label, pred_label):
    cm, _, _ = _contingency_matrix(true_label, pred_label)
    n = cm.sum()
    if n == 0:
        return 1.0

    pij = cm.astype(np.float64) / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)

    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / (pi @ pj)[nz]))

    pi1 = pi.ravel()
    pj1 = pj.ravel()
    h_true = -np.sum(pi1[pi1 > 0] * np.log(pi1[pi1 > 0]))
    h_pred = -np.sum(pj1[pj1 > 0] * np.log(pj1[pj1 > 0]))

    denom = h_true + h_pred
    if denom == 0:
        return 1.0
    return float(2.0 * mi / denom)


def _confusion_from_labels(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()]))
    else:
        labels = np.asarray(labels).ravel()
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def evaluate_labels(true_label, pred_label):
    """
    Evaluate clustering/classification labels.

    Returns a dict with:
      - ari
      - nmi
      - accuracy_aligned
      - average_accuracy_aligned
      - precision_aligned
      - recall_aligned
      - f1_aligned
      - macro_precision_aligned
      - macro_recall_aligned
      - macro_f1_aligned
      - weighted_precision_aligned
      - weighted_recall_aligned
      - weighted_f1_aligned
      - mapping (pred -> true)
      - pred_aligned
      - confusion_matrix_aligned
    """
    true_label = np.asarray(true_label).ravel()
    pred_label = np.asarray(pred_label).ravel()
    if true_label.shape != pred_label.shape:
        raise ValueError("true_label and pred_label must have the same shape")

    ari = _adjusted_rand_index(true_label, pred_label)
    nmi = _normalized_mutual_info(true_label, pred_label)

    pred_aligned, mapping = align_labels_hungarian(true_label, pred_label)
    # Keep union-based confusion matrix for reporting/visualization.
    cm_aligned, labels = _confusion_from_labels(true_label, pred_aligned)

    # Compute tp/fp/fn over true classes only (macro over true labels).
    true_classes = np.unique(true_label)
    tp = np.zeros(true_classes.size, dtype=np.float64)
    fp = np.zeros(true_classes.size, dtype=np.float64)
    fn = np.zeros(true_classes.size, dtype=np.float64)
    for i, c in enumerate(true_classes):
        yt = (true_label == c)
        yp = (pred_aligned == c)
        tp[i] = np.sum(yt & yp)
        fp[i] = np.sum((~yt) & yp)
        fn[i] = np.sum(yt & (~yp))

    total = true_label.size
    acc = float(np.mean(true_label == pred_aligned)) if total > 0 else 1.0

    # "ordinary" precision/recall/F1: micro-averaged over true classes
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()
    precision_micro = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    recall_micro = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    f1_micro = (
        float(2.0 * precision_micro * recall_micro / (precision_micro + recall_micro))
        if (precision_micro + recall_micro) > 0 else 0.0
    )

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    class_acc = recall.copy()
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0
    )
    support = tp + fn
    w = support / np.sum(support) if np.sum(support) > 0 else np.zeros_like(support)
    average_accuracy = float(np.mean(class_acc)) if class_acc.size else 1.0
    weighted_precision = float(np.sum(w * precision)) if precision.size else 1.0
    weighted_recall = float(np.sum(w * recall)) if recall.size else 1.0
    weighted_f1 = float(np.sum(w * f1)) if f1.size else 1.0

    return {
        "accuracy_aligned": acc,
        "weighted_precision_aligned": weighted_precision,
        "weighted_recall_aligned": weighted_recall,
        "weighted_f1_aligned": weighted_f1,
    }

def save_confusion_matrix_image(cm, labels=None, out_path="confusion_matrix.png", title="Confusion Matrix"):
    cm = np.asarray(cm)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # 在格子里写数值
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _masked_hamming_rate(x, y, missing_value=0):
    mask = (x != missing_value) & (y != missing_value)
    obs = int(np.sum(mask))
    if obs == 0:
        return np.nan, 0
    mism = int(np.sum(x[mask] != y[mask]))
    return mism / obs, obs


def evaluate_data_separability(
    X,
    true_label,
    missing_value=0,
    sample_pairs=20000,
    min_overlap=5,
    seed=0
):
    """
    Evaluate separability for categorical SNP matrix with missing values.

    Assumes values are in {1,2,3,4} and missing_value (default 0) means unobserved.
    All distances are computed only on overlapping observed positions.
    """
    X = np.asarray(X)
    y = np.asarray(true_label).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.shape[0] != X.shape[0]:
        raise ValueError("true_label length must match X rows")
    if sample_pairs <= 0:
        raise ValueError("sample_pairs must be positive")
    if min_overlap < 1:
        raise ValueError("min_overlap must be >= 1")

    n, m = X.shape
    classes, y_inv = np.unique(y, return_inverse=True)
    k = classes.size
    if k < 2:
        raise ValueError("Need at least 2 classes")

    idx_by_class = [np.where(y_inv == ci)[0] for ci in range(k)]
    counts = np.array([idx.size for idx in idx_by_class], dtype=np.int64)

    # Class centroids via observed-mode per position
    centroids = np.zeros((k, m), dtype=X.dtype)
    for ci, idx in enumerate(idx_by_class):
        Xi = X[idx]
        Pi = Xi != missing_value
        centroids[ci] = centroid_multiclass_mode(Xi, Pi, missing_val=missing_value)

    # Within-class mismatch rate to own centroid
    within_rates = []
    valid_within = 0
    for i in range(n):
        ci = y_inv[i]
        rate, obs = _masked_hamming_rate(X[i], centroids[ci], missing_value=missing_value)
        if obs >= min_overlap:
            within_rates.append(rate)
            valid_within += 1
    within_rate_mean = float(np.mean(within_rates)) if within_rates else np.nan

    # Between-class centroid mismatch rate
    between_centroid_rates = []
    for i in range(k):
        for j in range(i + 1, k):
            rate, obs = _masked_hamming_rate(centroids[i], centroids[j], missing_value=missing_value)
            if obs >= min_overlap:
                between_centroid_rates.append(rate)
    between_centroid_rate_mean = float(np.mean(between_centroid_rates)) if between_centroid_rates else np.nan

    # Nearest-centroid classification under masked distance
    pred = np.full(n, -1, dtype=np.int64)
    margins = []
    valid_pred = 0
    for i in range(n):
        dists = np.full(k, np.inf, dtype=np.float64)
        for ci in range(k):
            rate, obs = _masked_hamming_rate(X[i], centroids[ci], missing_value=missing_value)
            if obs >= min_overlap:
                dists[ci] = rate
        if np.any(np.isfinite(dists)):
            pred[i] = int(np.argmin(dists))
            valid_pred += 1
            own = dists[y_inv[i]]
            alt = np.min(np.delete(dists, y_inv[i])) if k > 1 else np.inf
            if np.isfinite(own) and np.isfinite(alt):
                margins.append(alt - own)

    centroid_acc = float(np.mean(pred[pred >= 0] == y_inv[pred >= 0])) if valid_pred > 0 else np.nan
    margin_mean = float(np.mean(margins)) if margins else np.nan
    margin_pos_ratio = float(np.mean(np.array(margins) > 0)) if margins else np.nan

    # Pair sampling: intra vs inter mismatch rate
    rng = np.random.default_rng(seed)
    valid_same_classes = [idx for idx in idx_by_class if idx.size >= 2]
    intra_rates = []
    inter_rates = []

    if valid_same_classes:
        while len(intra_rates) < sample_pairs:
            pool = valid_same_classes[rng.integers(len(valid_same_classes))]
            a, b = rng.choice(pool, size=2, replace=False)
            rate, obs = _masked_hamming_rate(X[a], X[b], missing_value=missing_value)
            if obs >= min_overlap:
                intra_rates.append(rate)

        while len(inter_rates) < sample_pairs:
            c1, c2 = rng.choice(k, size=2, replace=False)
            a = idx_by_class[c1][rng.integers(idx_by_class[c1].size)]
            b = idx_by_class[c2][rng.integers(idx_by_class[c2].size)]
            rate, obs = _masked_hamming_rate(X[a], X[b], missing_value=missing_value)
            if obs >= min_overlap:
                inter_rates.append(rate)

    intra_mean = float(np.mean(intra_rates)) if intra_rates else np.nan
    inter_mean = float(np.mean(inter_rates)) if inter_rates else np.nan
    gap = inter_mean - intra_mean if np.isfinite(intra_mean) and np.isfinite(inter_mean) else np.nan
    ratio = inter_mean / (intra_mean + 1e-12) if np.isfinite(intra_mean) and np.isfinite(inter_mean) else np.nan
    overlap_prob = float(np.mean(np.array(inter_rates) <= np.array(intra_rates))) if intra_rates and inter_rates else np.nan

    return {
        "n_samples": int(n),
        "n_sites": int(m),
        "n_classes": int(k),
        "class_counts": counts,
        "missing_rate": float(np.mean(X == missing_value)),
        "min_overlap": int(min_overlap),
        "valid_within_count": int(valid_within),
        "valid_centroid_pred_count": int(valid_pred),
        "within_rate_mean": within_rate_mean,
        "between_centroid_rate_mean": between_centroid_rate_mean,
        "centroid_assignment_acc": centroid_acc,
        "centroid_margin_mean": margin_mean,
        "centroid_margin_positive_ratio": margin_pos_ratio,
        "intra_rate_mean": intra_mean,
        "inter_rate_mean": inter_mean,
        "rate_gap": gap,
        "rate_ratio": ratio,
        "intra_inter_overlap_prob": overlap_prob,
    }
