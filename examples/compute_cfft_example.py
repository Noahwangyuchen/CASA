# This is an example to compute Cfft for the target model.
# Here we take the Wan2.1-T2V-14B model as the source model and Krea-Realtime as the target model.

from safetensors.torch import load_file, save_file
import utils
import torch
import pickle

state_dict = load_file("/data/Krea/krea-realtime-video-14b.safetensors", device="cpu")

for k, v in state_dict.items():
    print(f"{k}: {v.shape}")

device = 'cpu'
for i in range(8):
    C_state_krea = {}
    filename = f"/data/svd_results/Wan2.1-T2V-14B-Diffusers_part{i}.pkl"
    svd_wan = utils.load_results(filename, device=device)
    print(f"SVD results for Wan2.1-T2V-14B-Diffusers_part{i} loaded")

    for k in svd_wan.keys():
        # Tip: You may need to change the key mapping according to your model structure
        if k not in state_dict.keys():
            raise KeyError(f"Key not found:{k}")
        print(f"Processing layer:{k}")
        U_wan = svd_wan[k]["U"].cuda().float()
        Vh_wan = svd_wan[k]["Vh"].cuda().float()
        S_wan = svd_wan[k]["S"].cuda().float()
        dim = S_wan.shape[0]
        W_wan = U_wan[:, :dim] @ torch.diag(S_wan) @ Vh_wan[:dim, :]
        W_krea = state_dict[k].cuda().float()
        W_diff_krea = W_krea - W_wan
        C_krea = U_wan.T @ W_diff_krea @ Vh_wan.T
        C_state_krea[k] = C_krea.to(torch.bfloat16).cpu()

    save_path_krea = f"/data/Cfft/Krea_part{i}.pkl"

    with open(save_path_krea, "wb") as f:
        pickle.dump(C_state_krea, f)
    print(f"Krea_part{i} saved")