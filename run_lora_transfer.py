#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file

import transfer
import utils


def parse_transfer_kwargs(raw: str | None) -> Dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --transfer-kwargs: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("--transfer-kwargs must be a JSON object")
    return data


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def transfer_single(
    transfer_module,
    lora_state: Dict[str, torch.Tensor],
    svd_src_path: str,
    cfft_path: str,
    load_device: str,
    method: str,
    ignore_keywords: List[str],
    transfer_kwargs: Dict,
) -> Dict[str, torch.Tensor]:
    print(f"[Load] SVD src: {svd_src_path}")
    svd_src = utils.load_results(svd_src_path, device=load_device)
    print(f"[Load] Cfft: {cfft_path}")
    dict_cfft = utils.load_results(cfft_path, device=load_device)

    with torch.no_grad():
        return transfer_module.transfer_lora(
            lora_state=lora_state,
            svd_src=svd_src,
            dict_Cfft=dict_cfft,
            transfer_method=method,
            ignore_keywords=ignore_keywords,
            transfer_kwargs=transfer_kwargs,
        )


def transfer_multipart(
    transfer_module,
    lora_state: Dict[str, torch.Tensor],
    svd_src_pattern: str,
    cfft_pattern: str,
    num_parts: int,
    load_device: str,
    method: str,
    ignore_keywords: List[str],
    transfer_kwargs: Dict,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for part in range(num_parts):
        svd_src_path = svd_src_pattern.format(part=part)
        cfft_path = cfft_pattern.format(part=part)
        print(f"[Part {part}] Loading:")
        print(f"  - {svd_src_path}")
        print(f"  - {cfft_path}")
        svd_src = utils.load_results(svd_src_path, device=load_device)
        dict_cfft = utils.load_results(cfft_path, device=load_device)

        with torch.no_grad():
            partial = transfer_module.transfer_lora(
                lora_state=lora_state,
                svd_src=svd_src,
                dict_Cfft=dict_cfft,
                transfer_method=method,
                ignore_keywords=ignore_keywords,
                transfer_kwargs=transfer_kwargs,
            )
        out.update(partial)
        del partial, svd_src, dict_cfft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LoRA transfer from notebook workflow as a standalone script."
    )
    parser.add_argument("--lora-path", required=True, help="Input LoRA safetensors path")
    parser.add_argument("--output-path", required=True, help="Output LoRA safetensors path")
    parser.add_argument(
        "--method",
        default="CASA",
        help="Transfer method used by transfer.transfer_lora (default: CASA)",
    )
    parser.add_argument(
        "--load-device",
        default="cpu",
        help="Device used when loading pickle SVD/Cfft (default: cpu)",
    )
    parser.add_argument(
        "--ignore-keyword",
        action="append",
        default=[],
        help="Keyword to skip layers; can be passed multiple times",
    )
    parser.add_argument(
        "--transfer-kwargs",
        default="",
        help='JSON object passed to transfer_lora, e.g. \'{"rotation_threshold":0.5,"q_threshold":0.3}\'',
    )

    single = parser.add_argument_group("single-file mode")
    single.add_argument("--svd-src-path", help="Source SVD pickle path")
    single.add_argument("--cfft-path", dest="cfft_path", help="Target Cfft pickle path")

    multi = parser.add_argument_group("multi-part mode")
    multi.add_argument(
        "--svd-src-pattern",
        help='SVD pattern, e.g. "/path/Wan_part{part}.pkl"',
    )
    multi.add_argument(
        "--cfft-pattern",
        dest="cfft_pattern",
        help='Cfft pattern, e.g. "/path/FastWan_part{part}.pkl"',
    )
    multi.add_argument(
        "--num-parts",
        type=int,
        default=0,
        help="Number of parts to process with pattern mode",
    )
    return parser


def validate_args(args: argparse.Namespace) -> str:
    has_single = bool(args.svd_src_path and args.cfft_path)
    has_multi = bool(args.svd_src_pattern and args.cfft_pattern and args.num_parts > 0)
    if has_single == has_multi:
        raise ValueError(
            "Choose exactly one mode: "
            "single-file (--svd-src-path + --cfft-path) or "
            "multi-part (--svd-src-pattern + --cfft-pattern + --num-parts)."
        )
    return "single" if has_single else "multi"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    mode = validate_args(args)
    transfer_kwargs = parse_transfer_kwargs(args.transfer_kwargs)

    print(f"[Load] LoRA: {args.lora_path}")
    lora_state = load_file(args.lora_path)

    if mode == "single":
        transferred = transfer_single(
            transfer_module=transfer,
            lora_state=lora_state,
            svd_src_path=args.svd_src_path,
            cfft_path=args.cfft_path,
            load_device=args.load_device,
            method=args.method,
            ignore_keywords=args.ignore_keyword,
            transfer_kwargs=transfer_kwargs,
        )
    else:
        transferred = transfer_multipart(
            transfer_module=transfer,
            lora_state=lora_state,
            svd_src_pattern=args.svd_src_pattern,
            cfft_pattern=args.cfft_pattern,
            num_parts=args.num_parts,
            load_device=args.load_device,
            method=args.method,
            ignore_keywords=args.ignore_keyword,
            transfer_kwargs=transfer_kwargs,
        )

    ensure_parent_dir(args.output_path)
    save_file(transferred, args.output_path)
    print(f"[Save] Transferred LoRA -> {args.output_path}")


if __name__ == "__main__":
    main()
