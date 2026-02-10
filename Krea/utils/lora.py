import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


def load_lora_state_dict(lora_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

    if lora_path.endswith(".safetensors") or lora_path.endswith(".sft"):
        if safe_load_file is None:
            raise ImportError("safetensors is required to load .safetensors LoRA files")
        state_dict = safe_load_file(lora_path, device="cpu")
    else:
        state_dict = torch.load(lora_path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(
        state_dict["state_dict"], dict
    ):
        state_dict = state_dict["state_dict"]
    return state_dict


def _group_lora_weights(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    layers: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, weight in state_dict.items():
        if key.endswith(".lora_A.weight"):
            prefix = key.replace(".lora_A.weight", "")
            layers.setdefault(prefix, {})["down"] = weight
        elif key.endswith(".lora_B.weight"):
            prefix = key.replace(".lora_B.weight", "")
            layers.setdefault(prefix, {})["up"] = weight
        elif ".lora_down.weight" in key:
            prefix = key.replace(".lora_down.weight", "")
            layers.setdefault(prefix, {})["down"] = weight
        elif ".lora_up.weight" in key:
            prefix = key.replace(".lora_up.weight", "")
            layers.setdefault(prefix, {})["up"] = weight
        elif key.endswith(".alpha") or key.endswith(".lora_alpha"):
            prefix = key.rsplit(".", 1)[0]
            layers.setdefault(prefix, {})["alpha"] = weight
    return layers


def _strip_known_prefixes(name: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _resolve_target_name(param_names: Iterable[str], base_name: str) -> Optional[str]:
    if base_name in param_names:
        return base_name
    weight_name = f"{base_name}.weight"
    if weight_name in param_names:
        return weight_name
    return None


def merge_lora_into_model(
    model: torch.nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    scale: float = 1.0,
    prefixes: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    if prefixes is None:
        prefixes = [
            "diffusion_model.",
            "model.",
            "generator.",
            "generator.model.",
            "module.",
        ]

    grouped = _group_lora_weights(lora_state_dict)
    params = dict(model.named_parameters())
    param_names = set(params.keys())
    applied: List[str] = []
    missing: List[str] = []

    for raw_key, weights in grouped.items():
        normalized_key = _strip_known_prefixes(raw_key, prefixes)
        target_name = _resolve_target_name(param_names, normalized_key)
        if target_name is None:
            missing.append(raw_key)
            continue

        if "down" not in weights or "up" not in weights:
            missing.append(raw_key)
            continue

        target_param = params[target_name]
        lora_down = weights["down"]
        lora_up = weights["up"]
        rank = lora_down.shape[0]
        alpha_tensor = weights.get("alpha")
        alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)
        scaling = scale * (alpha / rank)

        if lora_up.ndim == 4:
            lora_up = lora_up.squeeze(3).squeeze(2)
            lora_down = lora_down.squeeze(3).squeeze(2)
            lora_delta = torch.mm(lora_up, lora_down).unsqueeze(2).unsqueeze(3)
        else:
            lora_delta = torch.mm(lora_up, lora_down)

        if lora_delta.shape != target_param.data.shape:
            missing.append(raw_key)
            continue

        with torch.no_grad():
            target_param.add_(lora_delta.to(target_param.dtype) * scaling)
        applied.append(raw_key)

    return applied, missing
