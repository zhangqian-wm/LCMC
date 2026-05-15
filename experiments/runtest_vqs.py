import numpy as np
from pathlib import Path
from datetime import datetime

from lcmc import split_merge_refine
from utils import (
    assign_and_count,
    evaluate_labels,
    find_identical_nonzero_columns,
    restore_centers,
    split_rows_for_workers,
    strip_identical_columns,
)


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULT_ROOT = PROJECT_ROOT / f"results_{TIMESTAMP}"
SAVE_SPLIT_IDX = False


def run_demo():
    print(f"Saving outputs under: {RESULT_ROOT}")
    prefix_list = ["data_3_7strains", "data_2_5strains"]

    for prefix in prefix_list:
        file_stem = DATA_DIR / prefix
        strain_dir = RESULT_ROOT / prefix
        strain_dir.mkdir(parents=True, exist_ok=True)

        snvs_raw = np.load(str(file_stem) + "_read_matrices.npy")
        true_label = np.load(str(file_stem) + "_true_labels.npy")

        print("Number of SNVs:", snvs_raw.shape[0])
        print("Length of the original SNVs:", snvs_raw.shape[1])
        ident_cols, ident_vals, _ = find_identical_nonzero_columns(snvs_raw, missing_value=0)
        x_reduced, keep = strip_identical_columns(snvs_raw, ident_cols)
        print("Length of the SNVs after removing the identical part:", x_reduced.shape[1])
        print("Length after removing low coverage columns:", x_reduced.shape[1])

        print_metrics = True
        seeds = range(0, 20)
        num_subs_list = [5, 8]
        idx_parts = None

        for num_subs in num_subs_list:
            for seed in seeds:
                result_name = strain_dir / f"{TIMESTAMP}_{prefix}_num_subs{num_subs}_seed{seed}"
                result_name.parent.mkdir(parents=True, exist_ok=True)
                if num_subs > 1:
                    _, idx_parts = split_rows_for_workers(
                        x_reduced,
                        true_label,
                        n_workers=num_subs,
                        seed=seed,
                    )

                    if SAVE_SPLIT_IDX:
                        np.savez(
                            str(result_name) + "_idx_parts.npz",
                            **{f"part_{i}": part for i, part in enumerate(idx_parts)},
                        )

                recon_v, sm_info = split_merge_refine(
                    x_reduced,
                    idx_parts=idx_parts,
                    num_subs=num_subs,
                    return_info=True,
                    seed=seed,
                )

                v_full = restore_centers(recon_v, keep, ident_cols, ident_vals)
                np.save(str(result_name) + "_recon_V.npy", v_full)
                print("The estimated number of SNVs: ", recon_v.shape[0])

                print("Evaluating the performance...")
                predict_label = np.asarray(sm_info.get("final_label"))
                if predict_label.size == 0:
                    predict_label, _ = assign_and_count(x_reduced, recon_v)
                np.save(str(result_name) + "_predict_label.npy", predict_label)
                res = evaluate_labels(true_label, predict_label)
                res.update(sm_info)
                np.savez(str(result_name) + "_res.npz", **res)

                if print_metrics:
                    print(f"ARI: {float(res['ari']):.4f}")
                    print(f"NMI: {float(res['nmi']):.4f}")
                    print(f"Aligned Accuracy: {float(res['accuracy_aligned']):.4f}")
                    print(f"Average-Accuracy: {float(res['average_accuracy_aligned']):.4f}")
                    print(f"Weighted-Precision: {float(res['weighted_precision_aligned']):.4f}")
                    print(f"Weighted-Recall: {float(res['weighted_recall_aligned']):.4f}")
                    print(f"Weighted-F1: {float(res['weighted_f1_aligned']):.4f}")
                    print(f"Precision: {float(res['precision_aligned']):.4f}")
                    print(f"Recall: {float(res['recall_aligned']):.4f}")
                    print(f"F1: {float(res['f1_aligned']):.4f}")


if __name__ == "__main__":
    run_demo()
