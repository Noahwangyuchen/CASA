# Exploring Data-Free LoRA Transferability for Video Diffusion Models

This repository contains the implementation for the paper **Exploring Data-Free LoRA Transferability for Video Diffusion Models**.

This repository mainly focuses on the implementation of **CASA**, a **data-free** and **training-free** weight-space algorithm that transfers a LoRA trained on a **source model** to a **distilled target video diffusion model**.

## What CASA Uses

CASA relies on:

1. **SVD decomposition of the source model weights**.
2. **Routing Pattern (Cfft)** of target-model weight drift projected into the source model subspace.
3. The LoRA weights to be transferred.

Using these components, CASA computes layer-wise arbitration in spectral space and reconstructs transferable LoRA factors for the distilled model.

## Repository Layout

- `transfer.py`: CASA core algorithm and LoRA transfer entry (`transfer_lora`).
- `run_lora_transfer.py`: CLI script for transferring LoRA with CASA.
- `examples/compute_svd_example.py`: example for computing source-model SVD.
- `examples/compute_cfft_example.py`: example for computing target Routing Pattern (Cfft).
- `Krea/`: LoRA-enabled inference support for Krea and related optimizations (including model loading flow improvements).
- `RollingForcing/`: LoRA-enabled inference support for Rolling Forcing.

## Environment

Recommended Python: `>=3.10`

Install dependencies (adjust CUDA/PyTorch to your environment):

```bash
pip install torch safetensors scipy numpy
```

For `examples/*` scripts, you may also need:

```bash
pip install diffusers transformers accelerate
```

## Workflow

### 1. Compute source-model SVD

You may refer to:

- `examples/compute_svd_example.py`

This script demonstrates extracting per-layer `U/S/Vh` from source model weights and saving them as `.pkl` shards.

### 2. Compute target Routing Pattern (Cfft)

You may refer to:

- `examples/compute_cfft_example.py`

Given source SVD and target weights, this script computes projected weight drift (`Cfft`) in source subspace and stores it as `.pkl` shards.

### 3. Run CASA LoRA transfer

Use:

- `run_lora_transfer.py`

#### Single-file mode

```bash
python run_lora_transfer.py \
  --lora-path /path/to/input_lora.safetensors \
  --svd-src-path /path/to/source_svd.pkl \
  --cfft-path /path/to/target_cfft.pkl \
  --output-path /path/to/output_lora.safetensors \
  --method CASA \
  --transfer-kwargs '{"rotation_threshold":0.5,"q_threshold":0.3,"arbitrate_q":0.85,"target_rank":32}'
```

#### Multi-part mode

```bash
python run_lora_transfer.py \
  --lora-path /path/to/input_lora.safetensors \
  --svd-src-pattern "/path/to/source_part{part}.pkl" \
  --cfft-pattern "/path/to/target_part{part}.pkl" \
  --num-parts 8 \
  --output-path /path/to/output_lora.safetensors \
  --method CASA \
  --transfer-kwargs '{"rotation_threshold":0.5,"q_threshold":0.3,"arbitrate_q":0.85,"target_rank":32}'
```

## Notes
- You may need to adapt key mapping in example scripts depending on your model checkpoint naming.

## Krea and Rolling Forcing Support

This repository also includes LoRA-enabled inference support for the two distilled target models used in the paper:

- **Krea**
- **Rolling Forcing**

Please refer to subproject files in `Krea/` and `RollingForcing/` for model-specific inference commands and settings.

