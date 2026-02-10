import os
import argparse
import time
from typing import Optional

import torch
from torchvision.io import write_video
from omegaconf import OmegaConf
from einops import rearrange
import gradio as gr

from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline


# -----------------------------
# Globals (loaded once per process)
# -----------------------------
_PIPELINE: Optional[torch.nn.Module] = None
_DEVICE: Optional[torch.device] = None


def _ensure_gpu():
    if not torch.cuda.is_available():
        raise gr.Error("CUDA GPU is required to run this demo. Please run on a machine with an NVIDIA GPU.")
    # Bind to GPU:0 by default
    torch.cuda.set_device(0)


def _load_pipeline(
    config_path: str,
    checkpoint_path: Optional[str],
    use_ema: bool,
    lora_path: Optional[str],
    lora_scale: float,
    text_encoder_cpu_offload: bool,
    vae_cpu_offload: bool,
    vae_decode_on_cpu: bool,
) -> torch.nn.Module:
    global _PIPELINE, _DEVICE
    if _PIPELINE is not None:
        return _PIPELINE

    _ensure_gpu()
    _DEVICE = torch.device("cuda:0")

    # Load and merge configs
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    if text_encoder_cpu_offload:
        config.text_encoder_cpu_offload = True
    if vae_cpu_offload:
        config.vae_cpu_offload = True
    if vae_decode_on_cpu:
        config.vae_decode_on_cpu = True

    # Choose pipeline type based on config
    pipeline = CausalInferencePipeline(config, device=_DEVICE)


    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if use_ema and 'generator_ema' in state_dict:
            state_dict_to_load = state_dict['generator_ema']
            # Remove possible FSDP prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict_to_load.items():
                new_state_dict[k.replace("_fsdp_wrapped_module.", "")] = v
            state_dict_to_load = new_state_dict
        else:
            state_dict_to_load = state_dict.get('generator', state_dict)
        pipeline.generator.load_state_dict(state_dict_to_load, strict=False)

    if lora_path:
        from utils.lora import load_lora_state_dict, merge_lora_into_model
        lora_state = load_lora_state_dict(lora_path)
        applied, missing = merge_lora_into_model(
            pipeline.generator.model, lora_state, scale=lora_scale
        )
        print(f"LoRA merge applied: {len(applied)} layers, missing: {len(missing)}")

    # The codebase assumes bfloat16 on GPU
    pipeline = pipeline.to(device=_DEVICE, dtype=torch.bfloat16)
    if getattr(config, "text_encoder_cpu_offload", False):
        pipeline.text_encoder.to(torch.device("cpu"))
        pipeline.text_encoder.set_output_device(_DEVICE)
    if getattr(config, "vae_cpu_offload", False):
        pipeline.vae.to(torch.device("cpu"))
        pipeline.vae.set_output_device(
            _DEVICE,
            decode_on_cpu=getattr(config, "vae_decode_on_cpu", False)
        )
    pipeline.eval()

    # Quick sanity path check for Wan models to give friendly errors
    wan_dir = os.path.join('wan_models', 'Wan2.1-T2V-1.3B')
    if not os.path.isdir(wan_dir):
        raise gr.Error(
            "Wan2.1-T2V-1.3B not found at 'wan_models/Wan2.1-T2V-1.3B'.\n"
            "Please download it first, e.g.:\n"
            "huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B"
        )

    _PIPELINE = pipeline
    return _PIPELINE


def build_predict(
    config_path: str,
    checkpoint_path: Optional[str],
    output_dir: str,
    use_ema: bool,
    lora_path: Optional[str],
    lora_scale: float,
    text_encoder_cpu_offload: bool,
    vae_cpu_offload: bool,
    vae_decode_on_cpu: bool,
):
    os.makedirs(output_dir, exist_ok=True)

    def predict(prompt: str, num_frames: int) -> str:
        if not prompt or not prompt.strip():
            raise gr.Error("Please enter a non-empty text prompt.")

        num_frames = int(num_frames)
        if num_frames % 3 != 0 or not (21 <= num_frames <= 252):
            raise gr.Error("Number of frames must be a multiple of 3 between 21 and 252.")

        pipeline = _load_pipeline(
            config_path,
            checkpoint_path,
            use_ema,
            lora_path,
            lora_scale,
            text_encoder_cpu_offload,
            vae_cpu_offload,
            vae_decode_on_cpu,
        )

        # Prepare inputs
        prompts = [prompt.strip()]
        noise = torch.randn([1, num_frames, 16, 60, 104], device=_DEVICE, dtype=torch.bfloat16)

        torch.set_grad_enabled(False)
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            video = pipeline.inference_rolling_forcing(
                noise=noise,
                text_prompts=prompts,
                return_latents=False,
                initial_latent=None,
            )

        # video: [B=1, T, C, H, W] in [0,1]
        video = rearrange(video, 'b t c h w -> b t h w c')[0]
        video_uint8 = (video * 255.0).clamp(0, 255).to(torch.uint8).cpu()

        # Save to a unique filepath
        safe_stub = prompt[:60].replace(' ', '_').replace('/', '_')
        ts = int(time.time())
        filepath = os.path.join(output_dir, f"{safe_stub or 'video'}_{ts}.mp4")
        write_video(filepath, video_uint8, fps=16)
        print(f"Saved generated video to {filepath}")

        return filepath

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/rolling_forcing_dmd.yaml',
                        help='Path to the model config')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/rolling_forcing_dmd.pt',
                        help='Path to rolling forcing checkpoint (.pt). If missing, will run with base weights only if available.')
    parser.add_argument('--output_dir', type=str, default='videos/gradio', help='Where to save generated videos')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA weights when loading checkpoint')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA checkpoint (.pt/.bin/.safetensors)')
    parser.add_argument('--lora_scale', type=float, default=1.0, help='LoRA scale for merging into weights')
    parser.add_argument('--text_encoder_cpu_offload', action='store_true',
                        help='Keep text encoder on CPU to save VRAM')
    parser.add_argument('--vae_cpu_offload', action='store_true',
                        help='Keep VAE on CPU to save VRAM (slower)')
    parser.add_argument('--vae_decode_on_cpu', action='store_true',
                        help='When VAE is offloaded, keep decoded video on CPU to reduce VRAM')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Gradio server host')
    parser.add_argument('--server_port', type=int, default=7860, help='Gradio server port')
    args = parser.parse_args()

    predict = build_predict(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        use_ema=not args.no_ema,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        text_encoder_cpu_offload=args.text_encoder_cpu_offload,
        vae_cpu_offload=args.vae_cpu_offload,
        vae_decode_on_cpu=args.vae_decode_on_cpu,
    )

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(label="Text Prompt", lines=2, placeholder="A cinematic shot of a girl dancing in the sunset."),
            gr.Slider(label="Number of Latent Frames", minimum=21, maximum=252, step=3, value=21),
        ],
        outputs=gr.Video(label="Generated Video", format="mp4"),
        title="Rolling Forcing: Autoregressive Long Video Diffusion in Real Time",
        description=(
            "Enter a prompt and generate a video using the Rolling Forcing pipeline.\n"
            "**Note:** although Rolling Forcing generates videos autoregressivelty, current Gradio demo does not support streaming outputs, so the entire video will be generated before it is displayed.\n"
            "\n"
            "If you find this demo useful, please consider giving it a ⭐ star on [GitHub](https://github.com/TencentARC/RollingForcing)--your support is crucial for sustaining this open-source project. "
            "You can also dive deeper by reading the [paper](https://arxiv.org/abs/2509.25161) or exploring the [project page](https://kunhao-liu.github.io/Rolling_Forcing_Webpage) for more details."
        ),
        allow_flagging='never',
    )

    try:
        # Gradio <= 3.x
        demo.queue(concurrency_count=1, max_size=2)
    except TypeError:
        # Gradio >= 4.x
        demo.queue(max_size=2)
    demo.launch(server_name=args.server_name, server_port=args.server_port, show_error=True)


if __name__ == "__main__":
    main()
