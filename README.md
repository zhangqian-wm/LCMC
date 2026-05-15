# LCMC: Latent Structural Categorical Matrix Completion
 
> **Latent Structural Categorical Matrix Completion with Application to Quasispecies Analysis**
> Qian Zhang · Meixia Lin
> 2026.
 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Last updated](https://img.shields.io/badge/Last%20updated-May%202026-lightgrey)

 
---

## Overview

**LCMC** is a double-loop optimization framework for completing matrices with categorical entries. It is designed for discrete, non-ordinal variables and represents categorical observations through a structure-preserving latent factorization.

The method estimates an interpretable factor matrix and the corresponding sample assignments. Given an input matrix $R$, LCMC learns an approximation of the form: $R \approx U V^\top$, where $U$ encodes sample assignments to latent components, and $V$ is a categorical factor matrix whose rows describe the characteristic profile of each component.

In the viral quasispecies reconstruction setting, each latent component corresponds to a candidate viral strain and its nucleotide composition.

**Input**

- $R$: categorical matrix with missing or partially observed entries

**Output**

- predicted labels corresponding to $U$
- $V$: reconstructed categorical factor matrix

The completed matrix $\hat{R}=UV^\top$ is not saved directly. It can be recovered by selecting rows from $V$ according to the predicted labels.
 
---


## Repository Structure

```text
LCMC/
|-- data/
|   |-- data_2_5strains_read_matrices.npy      # downloaded separately
|   |-- data_2_5strains_true_labels.npy        # downloaded separately
|   |-- data_3_7strains_read_matrices.npy      # downloaded separately
|   |-- data_3_7strains_true_labels.npy        # downloaded separately
|-- experiments/
|   |-- runtest_vqs.py
|   |-- test.py
|-- src/
|   |-- lcmc.py
|   |-- utils.py
|-- README.md
`-- requirements.txt
```

## Data Availability

The full experimental data files are not included in this repository because of their size. They are hosted separately on Dropbox.

Download the data files from Dropbox and place them under the `data/` directory with the following names:

```text
data/
|-- data_2_5strains_read_matrices.npy
|-- data_2_5strains_true_labels.npy
|-- data_3_7strains_read_matrices.npy
`-- data_3_7strains_true_labels.npy
```

Dropbox folder: [download the data files](https://www.dropbox.com/scl/fo/s51czbt7xx3xjq7rk28m7/AK0CwqEamYp1YitROjVRaH4?rlkey=1zx9cqcuzahqz2cl5fi7wb7xy&st=nv56eek9&dl=0)

After downloading, the experiment scripts can be run without changing the file paths.

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Code
 
To verify that the core components are working correctly, run the minimal functional test.
 
```bash
python experiments/runtest.py
```
 
To reproduce the main experiment, Viral Quasispecies Reconstruction in Section 5.2.3, from the paper, which exercises all components of the method including the enhanced strategy. 
 
```bash
python experiments/runtest_vqs.py
``` 

This may take a while to complete. For a quick functional check, use `runtest.py` above instead.



---
## Output File and Evaluation Metrics
Each run writes results to a timestamped directory (e.g. `results_20260420_123456/`), including 
```text
results_20260420_123456/
|-- data_filename/
|   |-- *_predict_label.npy
|   |-- *_recon_V.npy
|   |-- *_res.npz
```
The categorical factor matrix and the predicted labels are stored in `*_recon_V.npy` and `*_predict_label.npy`, respectively. 
The completed matrix $\hat{R}$ can be recovered by selecting rows from `*_recon_V.npy` according to `*_predict_label.npy`.
A dictionary of the performance metrics is stored in `*_res.npz`, including 
 - aligned accuracy (`accuracy_aligned`)
 - weighted precision (`weighted_precision_aligned`)
 - weighted recall (`weighted_recall_aligned`)
 - weighted F1 score (`weighted_f1_aligned`)
   
Please refer to the paper for the definition of these metrics.

---

 
## Citation
 
If you find this work useful, please cite:
 
```bibtex
@article{zhang2024lcmc,
  title   = {Latent Structural Categorical Matrix Completion with Application to Quasispecies Analysis},
  author  = {Zhang, Qian and Lin, Meixia},
  journal = {},
  year    = {2026}
}
```
 
---
 
## License
 
This project is released under the [GNU General Public License v3.0](LICENSE).
