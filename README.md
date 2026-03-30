<div align="center">

# Exploring Data-Free LoRA Transferability for Video Diffusion Models

**CASA: a data-free, training-free method for transferring LoRA between video diffusion models**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#-installation)
[![Method](https://img.shields.io/badge/Method-CASA-orange)](#-what-is-casa)
[![Scope](https://img.shields.io/badge/Target-Video%20Diffusion-green)](#-quick-start)

</div>
 
## 💡 TL;DR

This repository focuses on **CASA**, a practical pipeline to transfer a LoRA trained on a **source model** (e.g., Wan 2.1 14B) to a **distilled target model** (e.g., Krea / Rolling Forcing) **without extra data or finetuning**.

- No target-domain training data required.
- No additional optimization loop required.
- Works directly in spectral/weight space from model weights + LoRA.

## 🧠 What Is CASA

CASA (Cluster-Aware Spectral Arbitration) transfers LoRA by combining:

1. Source-model SVD basis (`U/S/Vh`).
2. Target Routing Pattern (`Cfft`) projected in source subspace.
3. Original LoRA factors (`A/B` or `down/up`).

It performs layer-wise spectral arbitration and reconstructs transferred LoRA factors for the target model.

## 📁 Repository Layout

- `transfer.py`: core CASA algorithm and `transfer_lora` entry.
- `run_lora_transfer.py`: CLI for single-file and multi-part transfer.
- `utils.py`: I/O, key mapping, and helper utilities.
- `examples/compute_svd_example.py`: example for source SVD extraction.
- `examples/compute_cfft_example.py`: example for target `Cfft` computation.
- `Krea/`: target-side inference integration for Krea.
- `RollingForcing/`: target-side inference integration for Rolling Forcing.

## 🛠️ Installation

Recommended Python: `3.10+`

```bash
pip install torch safetensors scipy numpy
```

For scripts under `examples/`:

```bash
pip install diffusers transformers accelerate
```

## 🚀 Quick Start

### 1) Prepare source SVD and target Cfft

Use the provided examples as templates:

- `examples/compute_svd_example.py`
- `examples/compute_cfft_example.py`

### 2) Run LoRA transfer (single-file mode)

```bash
python run_lora_transfer.py \
  --lora-path /path/to/input_lora.safetensors \
  --svd-src-path /path/to/source_svd.pkl \
  --cfft-path /path/to/target_cfft.pkl \
  --output-path /path/to/output_lora.safetensors \
  --method CASA \
  --transfer-kwargs '{"rotation_threshold":0.5,"q_threshold":0.3,"arbitrate_q":0.85,"target_rank":32}'
```

### 3) Run LoRA transfer (multi-part mode)

```bash
python run_lora_transfer.py \
  --lora-path /path/to/input_lora.safetensors \
  --svd-src-pattern '/path/to/source_part{part}.pkl' \
  --cfft-pattern '/path/to/target_part{part}.pkl' \
  --num-parts 8 \
  --output-path /path/to/output_lora.safetensors \
  --method CASA \
  --transfer-kwargs '{"rotation_threshold":0.5,"q_threshold":0.3,"arbitrate_q":0.85,"target_rank":32}'
```

## ⚙️ Important CLI Arguments

- `--method`: transfer method name (default `CASA`).
- `--load-device`: device used to load SVD/Cfft pickle files (default `cpu`).
- `--ignore-keyword`: skip matching LoRA layers (repeatable).
- `--transfer-kwargs`: JSON for CASA hyperparameters.
- `--svd-src-path` + `--cfft-path`: single-file mode.
- `--svd-src-pattern` + `--cfft-pattern` + `--num-parts`: multi-part mode.

## 🔌 Use Transferred LoRA in Target Projects

After generating `output_lora.safetensors`, run model-side inference in:

- `Krea/README.md`
- `RollingForcing/README.md`

Both subprojects already include LoRA loading options in their inference flows.

## 📝 Notes

- Key naming may differ across checkpoints; update mapping logic if needed.
- `examples/` scripts are intentionally minimal and should be adapted to your storage paths and model structure.
- Transfer currently expects CUDA availability during CASA computation in `transfer.py`.

## 🙏 Acknowledgements

- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan](https://github.com/Wan-Video/Wan2.1)
