import numpy as np
import scipy.sparse.linalg
import time

from utils import proj_vh, mat2ten, ten2mat, split_rows_for_workers, assign_and_count, compute_MEC, sorted_svds,centroid_multiclass_mode
from utils import compute_SSE

def tensor_factorization(M, P, M_ten, P_ten, k, SVD_S, SVD_Vt, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    M_ten = np.array(M_ten, dtype=np.float32)
    P_ten = P_ten.astype(bool)
    rows, cols = np.where(~P_ten)
    n, m_mat = M.shape
    sse = n * m_mat * 10 ** 12
    vh_init = np.sqrt(SVD_S[:k]).reshape(k, 1) * SVD_Vt[:k, :]

    for svd_flag in range(2):
        if svd_flag:
            vh_init = - vh_init
        vh = proj_vh(vh_init)

        # Solve U and V
        ite = 0
        fun = 1e10
        fun_err = 1
        vh_err = 1
        maxit = 10 ** 4
        eps = 10 ** (-4)
        min_index, fre = assign_and_count(M, ten2mat(vh))

        while (ite < maxit) & (vh_err > eps) & (fun_err > eps) & (fun > eps):  #
            vh_old = vh.copy()
            fun_old = fun
            xfill = M_ten.copy()
            xfill[rows, cols] = vh[min_index[rows], cols]

            # update V
            vh_sum = np.zeros_like(vh)
            np.add.at(vh_sum, min_index, xfill)
            nonempty = fre > 0
            vh[nonempty] = vh_sum[nonempty] / fre[nonempty, None]
            empty = np.where(fre == 0)[0]
            if empty.size > 0:
                rand_rows = np.random.choice(M_ten.shape[0], size=empty.size, replace=False)
                vh[empty] = M_ten[rand_rows]

            min_index, fre = assign_and_count(M, ten2mat(proj_vh(vh)))
            fun = 0.5 * compute_SSE(M_ten,vh,label=min_index,P=P_ten)
            fun_err = abs(fun - fun_old) / max(1, fun_old)
            vh_err = np.linalg.norm(vh - vh_old, 'fro') / m_mat / k
            ite = ite + 1

        vh = proj_vh(vh)
        vh = ten2mat(vh)

        true_ind, fre = assign_and_count(M, vh)
        # tmp_cri = compute_MEC(M, vh, P=P, label=true_ind) # len(np.where((R - M) != 0)[0])
        tmp_cri = compute_SSE(M, vh, P=P, label=true_ind) # len(np.where((R - M) != 0)[0])

        if tmp_cri < sse:
            sse = tmp_cri
            reconV = vh
            output_ind = true_ind
        print('------ Number of iterations: ' + str(ite))

    return reconV, sse, output_ind


def reconstruct_haplotype(SNVmatrix, return_label=False, k_max=None, svd_mode="per_k", seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)

    (num_read, m_mat) = SNVmatrix.shape  # number of reads, length of haplotypes
    M = SNVmatrix.copy()  # initial read matrix
    P = (SNVmatrix != 0)  # projection matrix
    P_tensor_unfold = np.tile(P[:, :, np.newaxis], (1, 4)).reshape(num_read, -1)
    M_tensor_unfold = mat2ten(M)

    k_flag = 0
    k_upper = num_read - 1
    eps = 0.01
    if k_max is not None:
        k_upper = min(k_upper, int(k_max))
    if k_upper < 1:
        raise ValueError(f"k_max is too small: k_upper={k_upper}, num_read={num_read}")

    k_init = min(3, k_upper)
    k_search_thre = 1 - eps

    SVD_U, SVD_S, SVD_Vt = sorted_svds(M_tensor_unfold, k=k_init, rng=rng)

    k_search = k_init
    k_table = np.array([1, k_upper + 1])  # tracking Kmin and Kmax (exclusive upper bound)

    s1 = time.time()
    problem_total_time = 0
    SSE = compute_MEC(M, ten2mat(proj_vh(np.mean(M_tensor_unfold, axis=0).reshape((1, -1)))), P)
    if svd_mode not in {"per_k", "max_cached"}:
        raise ValueError("svd_mode must be one of {'per_k', 'max_cached'}")

    k_hist = {}
    # SVD cache modes:
    # per_k: store decomposition for each exact k.
    # max_cached: keep only the current largest rank and slice for smaller k.
    svd_hist = {k_init: (SVD_S, SVD_Vt)} if svd_mode == "per_k" else {}
    cached_rank = k_init if svd_mode == "max_cached" else None
    cached_S = SVD_S if svd_mode == "max_cached" else None
    cached_Vt = SVD_Vt if svd_mode == "max_cached" else None
    Vh_record = None
    while (k_table[1] - k_table[0] > 1):
        criterion = np.zeros(2)
        print('-- k_search: ' + str(k_search))
        k_candidates = [k_search]
        if k_search + 1 <= k_upper:
            k_candidates.append(k_search + 1)
        k_hi = max(k_candidates)

        # One-shot SVD prefetch for current k neighborhood.
        # This avoids computing SVD twice for k_search and k_search+1.
        if svd_mode == "per_k" and k_search not in svd_hist:
            if k_hi not in svd_hist:
                _, S_hi, Vt_hi = sorted_svds(M_tensor_unfold, k=k_hi, rng=rng)
                svd_hist[k_hi] = (S_hi, Vt_hi)
            S_hi, Vt_hi = svd_hist[k_hi]
            svd_hist[k_search] = (S_hi[:k_search], Vt_hi[:k_search, :])

        if svd_mode == "max_cached":
            if (cached_rank is None) or (cached_rank < k_hi):
                _, cached_S, cached_Vt = sorted_svds(M_tensor_unfold, k=k_hi, rng=rng)
                cached_rank = k_hi

        for k in k_candidates:
            sub_time = time.time()
            if k in k_hist:
                Vh, SSE, ind = k_hist[k]
            else:
                if svd_mode == "per_k":
                    if k not in svd_hist:
                        _, S_tmp, Vt_tmp = sorted_svds(M_tensor_unfold, k=k, rng=rng)
                        svd_hist[k] = (S_tmp, Vt_tmp)
                    SVD_S, SVD_Vt = svd_hist[k]
                elif svd_mode == "max_cached":
                    if (cached_rank is not None) and (k <= cached_rank):
                        SVD_S = cached_S[:k]
                        SVD_Vt = cached_Vt[:k, :]
                    else:
                        _, cached_S, cached_Vt = sorted_svds(M_tensor_unfold, k=k, rng=rng)
                        cached_rank = k
                        SVD_S, SVD_Vt = cached_S, cached_Vt

                # factorization
                Vh, SSE, ind = tensor_factorization(
                    M, P, M_tensor_unfold, P_tensor_unfold, k, SVD_S, SVD_Vt, rng=rng
                )
                k_hist[k] = (Vh, SSE, ind)

            if k == k_search:
                V1 = Vh.copy()
                SSE_old = SSE
            print('---- k: ' + str(k) + '\t' + 'SSE: ' + str(SSE) + '\t' + 'TRC time: ' + str(time.time() - sub_time))
            criterion[k - k_search] = SSE

        ## Compute SSE_ratio
        SSE_ratio = SSE/SSE_old
        print('---- sse_ratio: ' +str(SSE_ratio))

        if SSE_ratio > k_search_thre:
            k_table[1] = k_search
            k_search = np.floor((k_search + max(1, k_table[0])) / 2)
            k_flag = 1
            Vh_record = np.array(V1, dtype=int)
        else:
            if k_flag == 0:
                k_table[0] = k_search
                k_search = 2 * k_search
            else:
                k_table[0] = k_search
                k_search = np.floor((k_search + k_table[1]) / 2)
        k_search = int(k_search)
        if k_search > k_upper:
            k_search = k_upper

        # Prune caches: keep only keys inside [k_table[0], k_table[1] + 1]
        # (integer keys => keep up to k_table[1] to cover k_search+1 usage).
        active_lo = max(1, int(k_table[0]))
        active_hi = min(k_upper, int(k_table[1]))
        active_keys = set(range(active_lo, active_hi + 1))

        for kk in list(k_hist.keys()):
            if kk not in active_keys:
                del k_hist[kk]

        if svd_mode == "per_k":
            for kk in list(svd_hist.keys()):
                if kk not in active_keys:
                    del svd_hist[kk]


    s2 = time.time() - s1
    problem_total_time = problem_total_time + s2

    if Vh_record is None:
        Vh_record = np.array(V1, dtype=int)
    reconV = np.array(Vh_record, dtype=int)
    print('-- k_opt: ' + str(reconV.shape[0]) + ';' + ' LCMC time: ' + str(problem_total_time))

    results = {'time': problem_total_time}
    if return_label:
        label,_ = assign_and_count(M, reconV)
        return reconV, label, results
    else:
        return reconV, results


def split_merge_refine(
    SNVs,
    idx_parts=None,
    num_subs=5,
    k_max=50,
    svd_mode="max_cached",
    return_info=False,
    clip_solution=False,
    final_freq_threshold=0.01,
    seed=None,
):
    rng_main = np.random.default_rng(seed)
    t_total_start = time.perf_counter()
    subproblem_times = []
    print('Divide into ' + str(num_subs) + ' sub-problems')
    if num_subs == 1:
        t_split_start = time.perf_counter()
        sub_seed = int(rng_main.integers(0, 2**32 - 1))
        Vs, sub_res = reconstruct_haplotype(SNVs, k_max=k_max, svd_mode=svd_mode, seed=sub_seed)
        subproblem_times.append(float(sub_res.get("time", 0.0)))
        split_time = time.perf_counter() - t_split_start
        merge_time = 0.0
        refine_time = 0.0
    else:
        t_split_start = time.perf_counter()
        if idx_parts is None:
            raise ValueError("idx_parts not specified")
        elif len(idx_parts) != num_subs:
            raise ValueError("num_subs not equal to len(idx_parts)")
        V_subs = np.empty(num_subs, dtype=object)
        for ind_sub in range(num_subs):
            print('-------------------')
            print('Solving subproblem: ' + str(ind_sub+1) +' / ' + str(num_subs))
            sub_seed = int(rng_main.integers(0, 2**32 - 1))
            V_subs[ind_sub], sub_res = reconstruct_haplotype(
                SNVs[idx_parts[ind_sub]], return_label=False, k_max=k_max, svd_mode=svd_mode, seed=sub_seed
            )
            subproblem_times.append(float(sub_res.get("time", 0.0)))
        Vs = np.concatenate(V_subs, axis=0)
        P = (SNVs != 0)
        split_time = time.perf_counter() - t_split_start

        label, freq = assign_and_count(SNVs, Vs)

        # remove empty clusters (if any)
        nonempty = freq > 0
        k = Vs.shape[0]
        if not np.all(nonempty):
            Vs = Vs[nonempty]
            label_map = np.full(k, -1, dtype=np.int64)
            label_map[np.where(nonempty)[0]] = np.arange(nonempty.sum())
            label = label_map[label]
            k = Vs.shape[0]
            freq = np.bincount(label, minlength=k)

        # merge
        t_merge_start = time.perf_counter()
        print('Start merging with k=' + str(Vs.shape[0]) )
        # Vs changes during merge, so iterate with dynamic indices.
        i = 0
        while i < Vs.shape[0]:
            if freq[i] == 0:
                i += 1
                continue

            rows_i = (label == i)
            Xi, Pi = SNVs[rows_i], P[rows_i]
            mec_i = compute_MEC(Xi, Vs[i], Pi)

            merged_i = False
            j = i + 1
            while j < Vs.shape[0]:
                if freq[j] == 0:
                    j += 1
                    continue

                rows_j = (label == j)
                Xj, Pj = SNVs[rows_j], P[rows_j]
                mec_j = compute_MEC(Xj, Vs[j], Pj)

                # merged centroid from combined samples (binary mode on observed)
                Xij = np.vstack([Xi, Xj])
                Pij = np.vstack([Pi, Pj])
                centroid = centroid_multiclass_mode(Xij, Pij, missing_val=0)

                mec_cent = compute_MEC(Xij, centroid, Pij)

                # merge if total mismatch decreases
                if mec_cent < (mec_i + mec_j):
                    # perform merge: replace i with centroid, delete j
                    Vs[i] = centroid
                    Vs = np.delete(Vs, j, axis=0)

                    # re-assign labels after topology changes, keep i and continue.
                    label, freq = assign_and_count(SNVs, Vs)
                    rows_i = (label == i)
                    Xi, Pi = SNVs[rows_i], P[rows_i]
                    mec_i = compute_MEC(Xi, Vs[i], Pi)
                    merged_i = True
                    continue
                j += 1
            if not merged_i:
                i += 1
        merge_time = time.perf_counter() - t_merge_start

        #refine
        print('Start refining with k=' + str(Vs.shape[0]))
        t_refine_start = time.perf_counter()
        while Vs.shape[0]>1:
            label, fre = assign_and_count(SNVs, Vs)
            sse_before = np.array(SNVs != Vs[label,:]).sum()

            # Identify the group with the lowest frequency (use U to determine the group frequencies)
            group_to_remove = np.argmin(fre)  # Identify the group with the lowest frequency

            # Remove the strain corresponding to the group with the lowest frequency
            refined_V = np.delete(Vs, group_to_remove, axis=0)

            # Compute the sum of squared errors (SSE) after refinement
            label, fre = assign_and_count(SNVs, refined_V)
            sse_after = np.array(SNVs != refined_V[label,:]).sum()

            # print('SSE before: ' + str(sse_before))
            # print('SSE after: ' + str(sse_after))

            # Only accept the refinement if the SSE improvement is acceptable
            if sse_after <= sse_before:
                 Vs = refined_V
            else:
                break  # Return the original tensor if no improvement
        refine_time = time.perf_counter() - t_refine_start

        if clip_solution:
            label, fre = assign_and_count(SNVs, Vs)
            Vs = Vs[fre>=0.05,:]

    final_label, final_freq = assign_and_count(SNVs, Vs)
    final_freq_ratio = final_freq / max(len(final_label), 1)
    num_groups_before_threshold = int(Vs.shape[0])
    if final_freq_threshold is not None and final_freq_threshold > 0:
        keep = final_freq_ratio >= float(final_freq_threshold)
        if not np.any(keep):
            keep[np.argmax(final_freq)] = True
        if not np.all(keep):
            removed = np.where(~keep)[0].tolist()
            print(
                f"Truncate low-frequency groups (< {float(final_freq_threshold):.4f}): "
                f"remove {removed}"
            )
            Vs = Vs[keep]
            final_label, final_freq = assign_and_count(SNVs, Vs)
            final_freq_ratio = final_freq / max(len(final_label), 1)

    total_time = time.perf_counter() - t_total_start
    print(f"[timing] split: {split_time:.4f}s, merge: {merge_time:.4f}s, refine: {refine_time:.4f}s, total: {total_time:.4f}s")
    print(f"[timing] subproblems: {[round(t, 4) for t in subproblem_times]}")

    info = {
        "split_time": float(split_time),
        "merge_time": float(merge_time),
        "refine_time": float(refine_time),
        "total_time": float(total_time),
        "subproblem_times": np.array(subproblem_times, dtype=float),
        "num_subs": int(num_subs),
        "final_freq_threshold": float(final_freq_threshold),
        "num_groups_before_threshold": int(num_groups_before_threshold),
        "num_groups_after_threshold": int(Vs.shape[0]),
        "final_label": np.asarray(final_label, dtype=np.int64),
        "final_freq": np.asarray(final_freq, dtype=np.int64),
        "final_freq_ratio": np.asarray(final_freq_ratio, dtype=float),
    }


    if return_info:
        return Vs, info
    return Vs


if __name__ == "__main__":

    np.random.seed(1)
    file = '../data/simu_5strains_read_matrix.txt'
    SNVmatrix = np.loadtxt(file)
    recon_V = reconstruct_haplotype(SNVmatrix)
    # np.savetxt('../results/recon_V.txt', recon_V, fmt='%d')

    # SNVs = np.load("../data/real_5strains_read_matrix.npy", allow_pickle=True)
    # recon_V = split_merge_refine(SNVs)
    # print(recon_V.shape)
    # # np.savetxt('../results/recon_V.txt', recon_V, fmt='%d')


