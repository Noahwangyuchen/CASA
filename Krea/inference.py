import argparse
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from release_server import GenerateParams
from sample import sample_videos

# Configure generation parameters
params = GenerateParams(
    prompt="",  # Will be overwritten per prompt
    width=832,
    height=480,
    num_blocks=9,
    seed=42,
    kv_cache_num_frames=3,
)

# Define prompts
prompts = [
    "A hyperrealistic close-up of ocean waves shimmering at sunset.",
    "A bustling neon-drenched alleyway with rain-soaked pavement.",
]

def _maybe_write_temp_config(config_path: str, lora_path: str | None, lora_scale: float) -> str:
    if not lora_path:
        return config_path

    cfg = OmegaConf.load(config_path)
    cfg.lora_path = lora_path
    cfg.lora_scale = lora_scale
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    tmp.close()
    OmegaConf.save(cfg, tmp.name)
    return tmp.name


def _print_config_summary(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    keys = [
        "checkpoint_path",
        "lora_path",
        "lora_scale",
        "skip_base_weights",
        "timestep_shift",
        "num_frame_per_block",
        "num_train_timestep",
        "guidance_scale",
        "enable_fp8",
    ]
    print("Config summary:")
    for key in keys:
        value = cfg.get(key, None)
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="/workspace/usr/yuchen/krea/configs/self_forcing_server_14b.yaml")
    parser.add_argument("--output_dir", default="outputs/samples")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    args = parser.parse_args()

    temp_config_path = _maybe_write_temp_config(args.config_path, args.lora_path, args.lora_scale)
    try:
        _print_config_summary(temp_config_path)
        sample_videos(
            prompts_list=prompts,
            config_path=temp_config_path,
            output_dir=args.output_dir,
            params=params,
            save_videos=True,  # Requires ffmpeg
            fps=args.fps,
        )
    finally:
        if temp_config_path != args.config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()
