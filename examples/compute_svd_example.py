# This is an example of doing SVD decomposition for the source model.
# Here we take the Wan2.1-T2V-14B model as an example.

import os
import torch
import gc
import pickle
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler


model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, device="cuda")
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, device="cuda")

transformer = pipe.transformer 

save_dir = "/data/svd_results/"
os.makedirs(save_dir, exist_ok=True)

svd_results = {}
part_id = 0
block_count = 0
device = torch.device("cuda:0")

for name, param in transformer.named_parameters():
    if not name.startswith("blocks."):
        continue

    # Extract block index, e.g. "blocks.12.attn1.to_q.weight" -> 12
    block_idx = int(name.split('.')[1])

    # When moving to the next block, update block_count
    if block_idx != block_count:
        # Every 5 blocks, write current results to disk
        if block_idx % 5 == 0:
            if svd_results:
                save_path = os.path.join(save_dir, f"Wan2.1-T2V-14B-Diffusers_part{part_id}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(svd_results, f)
                print(f"✅ Saved part {part_id} with {len(svd_results)} items -> {save_path}")

                part_id += 1
                svd_results.clear()
                gc.collect()

        block_count = block_idx

    # Only run SVD for 2D matrices
    if param.dim() == 2:
        print(name, param.shape)
        W = param.detach().float().to(device)
        U, S, Vh = torch.linalg.svd(W, full_matrices=True)

        # Move to CPU + bfloat16 to reduce storage
        svd_results[name] = {
            "U": U.to(torch.bfloat16),
            "S": S.to(torch.bfloat16),
            "Vh": Vh.to(torch.bfloat16),
        }

        # Release GPU memory
        del W, U, S, Vh
        torch.cuda.empty_cache()
        gc.collect()

# Save the last chunk (possibly <5 blocks)
if svd_results:
    save_path = os.path.join(save_dir, f"Wan2.1-T2V-14B-Diffusers_part{part_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(svd_results, f)
    print(f"✅ Saved final part {part_id} with {len(svd_results)} items -> {save_path}")
