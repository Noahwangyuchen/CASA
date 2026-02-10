import torch
import utils
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def cluster_aware_spectral_arbitration(B, A, 
                                  U_s, S_s, Vh_s, 
                                  Cfft,  
                                  rotation_threshold=0.5,
                                  q_threshold=0.5,
                                  arbitrate_q=0.85,
                                  target_rank=32):
    """
    CASA algorithm:
    1. Density stratification: identify Dominant and Non-Dominant regions.
    2. Hybrid scoring metric:
       - Score = Pixel_Energy * Context_Factor
       - Pixel_Energy = |Clora * Cfft|
       - Context_Factor:
         >> Inside Top-K: Block Cosine Similarity (cluster-level directional consistency)
         >> Outside Top-K: 1.0
    3. Strategy:
       - In Low Mask regions, directly use Clora - Cfft to preserve LoRA effect.
       - In High Mask regions with same-sign pixels, compute score quantiles.
       - Pixels with Score > Threshold are arbitrated as max(Clora, Cfft) - Cfft
         to prevent over-activation.
       - Others are set to Clora.
    """
    print(f"--- Starting CASA ---")
    
    # =========================================================
    # Step 0: Base definitions and projection
    # =========================================================
    device = B.device
    D_out, D_in = Cfft.shape
    
    M_left = U_s.t() @ B
    M_right = A @ Vh_s.t()
    Clora = M_left @ M_right

    if Clora.shape[0] == Clora.shape[1]: 
        energies = S_s ** 2
        cumulative_energy = torch.cumsum(energies, dim=0) / energies.sum()
        k_dynamic = torch.where(cumulative_energy >= 0.90)[0]
        k = k_dynamic[0].item() + 1 if len(k_dynamic) > 0 else S_s.size(0)
    else:
        k = S_s.size(0)
    
    # =========================================================
    # Step 1: Perturbation-based clustering
    # =========================================================
    Clora_k = Clora[:k, :k]
    S_k = S_s[:k]
    
    M_force = torch.abs(Clora_k)
    S_matrix = S_k.unsqueeze(1)
    Gap_matrix = torch.abs(S_matrix - S_matrix.t())
    M_predicted_rotation = M_force / (Gap_matrix + 1e-6)
    M_predicted_rotation.fill_diagonal_(0)
    
    rows_idx, cols_idx = torch.where(M_predicted_rotation > rotation_threshold)
    adj = np.zeros((k, k), dtype=int)
    r_np, c_np = rows_idx.cpu().numpy(), cols_idx.cpu().numpy()
    adj[r_np, c_np] = 1
    adj[c_np, r_np] = 1
    
    n_clusters, labels_np = connected_components(csr_matrix(adj), directed=False)
    labels = torch.tensor(labels_np, device=device)
    
    print(f"Identified {n_clusters} clusters.")

    # =========================================================
    # Step 2: Density filtering (single threshold q_threshold)
    # =========================================================
    Cfft_k = Cfft[:k, :k]
    M_mat = torch.nn.functional.one_hot(labels.long(), num_classes=n_clusters).float().t()
    Cluster_Sizes = M_mat.sum(dim=1) + 1e-6
    
    Rx_Density = (M_mat @ torch.norm(Cfft_k, dim=1)) / Cluster_Sizes
    Tx_Density = (M_mat @ torch.norm(Cfft_k, dim=0)) / Cluster_Sizes
    
    rx_th = torch.quantile(Rx_Density, q_threshold)
    tx_th = torch.quantile(Tx_Density, q_threshold)
    
    cl_rx_high = torch.where(Rx_Density >= rx_th)[0]
    cl_tx_high = torch.where(Tx_Density >= tx_th)[0]
    
    # Build High Mask (expanded to full matrix shape)
    idx_rx = torch.isin(labels, cl_rx_high)
    idx_tx = torch.isin(labels, cl_tx_high)
    full_rows = torch.zeros(D_out, dtype=torch.bool, device=device); full_rows[:k] = idx_rx
    full_cols = torch.zeros(D_in, dtype=torch.bool, device=device); full_cols[:k] = idx_tx
    mask_high = full_rows.unsqueeze(1) | full_cols.unsqueeze(0)

    # =========================================================
    # Step 3: Build hybrid score matrix
    # =========================================================
    
    # 1) Compute pixel-level energy
    pixel_energy_map = (Clora * Cfft).abs()
    
    # 2) Compute block-level cosine (Top-K only)
    dot_prod = M_mat @ (Clora_k * Cfft_k) @ M_mat.t()
    norm_clora = torch.sqrt(M_mat @ (Clora_k**2) @ M_mat.t())
    norm_cfft = torch.sqrt(M_mat @ (Cfft_k**2) @ M_mat.t())
    block_cosine = dot_prod / (norm_clora * norm_cfft + 1e-8)
    
    # 3) Map block cosine back to pixel space (context factor)
    # Initialize with 1.0 everywhere (outside Top-K defaults to multiplier 1)
    context_factor_map = torch.ones((D_out, D_in), device=device, dtype=Clora.dtype)
    
    # Fill Top-K region with corresponding block cosine values
    # M_mat.T [k, n] @ [n, n] @ [n, k] -> [k, k]
    pixel_level_block_cosine = M_mat.t() @ block_cosine @ M_mat
    context_factor_map[:k, :k] = pixel_level_block_cosine
    
    # 4) Final hybrid score
    # Score = Pixel_Energy * Context_Factor
    # Note: if block cosine is negative (direction conflict), the score is negative
    # and is naturally filtered out by quantile thresholding.
    hybrid_score = pixel_energy_map * context_factor_map

    # =========================================================
    # Step 4: Apply arbitration strategy
    # =========================================================
    mask_same = (Clora.sign() == Cfft.sign())
    
    Clora_new = Clora - Cfft
    
    Clora_new[mask_high] = Clora[mask_high]
    # 2) High & same-sign -> arbitrate using Hybrid Score
    mask_high_same = mask_high & mask_same
    
    # Extract scores from candidate region
    if mask_high_same.sum() > 0:
        candidate_scores = hybrid_score[mask_high_same]
        
        # Compute threshold
        num_elements = candidate_scores.numel()
        max_samples = 10_000_000
        if num_elements > max_samples:
            # Random sampling
            indices = torch.randperm(num_elements, device=candidate_scores.device)[:max_samples]
            sample_for_quantile = candidate_scores[indices]
            score_thresh = torch.quantile(sample_for_quantile, arbitrate_q)
        else:
            score_thresh = torch.quantile(candidate_scores, arbitrate_q)
        
        mask_suppress = mask_high_same & (hybrid_score > score_thresh)

        max_val = torch.where(Clora.abs() > Cfft.abs(), Clora, Cfft)
        C_arbitrated = max_val - Cfft
        Clora_new[mask_suppress] = C_arbitrated[mask_suppress]
        print(f"Suppressed {mask_suppress.sum().item()} pixels.")

    # =========================================================
    # Step 5: Reconstruct LoRA factors
    # =========================================================
    U_c, S_c, V_c = torch.svd_lowrank(Clora_new, q=target_rank, niter=6)
    Vh_c = V_c.t() 
    
    sqrt_S = torch.diag(torch.sqrt(S_c))
    B_new = U_s @ (U_c @ sqrt_S)
    A_new = (sqrt_S @ Vh_c) @ Vh_s
    
    return B_new, A_new


def transfer_lora(
    lora_state,
    svd_src,
    dict_Cfft,
    transfer_method,
    ignore_keywords=None,
    transfer_kwargs=None,
):
    """
    lora_state: LoRA parameter state dict.
    svd_src: source spectral basis dict[layer_name] -> {"U":..., "S":..., "Vh":...}.
    dict_Cfft: target Cfft map dict[layer_name] -> Tensor.
    transfer_method: transfer method (callable or predefined name).
    transfer_kwargs: kwargs forwarded to the selected transfer method.
    """

    new_state = {}
    processed = set()

    built_in_transfers = {
        "CASA": cluster_aware_spectral_arbitration,
    }
    if callable(transfer_method):
        transfer_fn = transfer_method
    elif isinstance(transfer_method, str) and transfer_method in built_in_transfers:
        transfer_fn = built_in_transfers[transfer_method]
    else:
        raise ValueError(f"Unknown transfer_method: {transfer_method}")
    print(f"Using transfer method: {transfer_fn.__name__}")

    # Forward kwargs (kept compatible with use_basis_transfer defaults)
    kwargs = dict(transfer_kwargs or {})

    for key in list(lora_state.keys()):
        if key in processed or (ignore_keywords and any(kw in key for kw in ignore_keywords)):
            continue
        
        # Detect LoRA A key
        if key.endswith("lora_A.weight"):
            prefix = key[:-len(".lora_A.weight")]
            key_A = key
            key_B = prefix + ".lora_B.weight"
        elif key.endswith("lora_down.weight"):
            prefix = key[:-len(".lora_down.weight")]
            key_A = key
            key_B = prefix + ".lora_up.weight"
        else:
            # Skip non-A/down weights for now; copy handling is done later
            continue

        print(f"Transferring LoRA as {prefix} ...")

        if key_B not in lora_state:
            # If paired B/up is missing, keep A unchanged
            print(f"[WARN] Pair not found for {key_A}, skip spectral transport.")
            new_state[key_A] = lora_state[key_A]
            continue

        processed.add(key_A)
        processed.add(key_B)

        # Infer base weight name, e.g. blocks.0.self_attn.q.weight
        svd_keys = list(svd_src.keys())  # or svd_wan_1_3.keys()
        base_weight_name = utils.auto_map_lora_key_to_svd_key(prefix, svd_keys)

        if base_weight_name not in svd_src:
            # print(f"[WARN] {base_weight_name} not found in SVD dicts, copy LoRA as-is.")
            # new_state[key_A] = lora_state[key_A]
            # new_state[key_B] = lora_state[key_B]
            print(f"[WARN] {base_weight_name} not found in SVD dicts, skip it.")
            continue

        # Load LoRA A/B
        A = lora_state[key_A]  # [r, n] or [r, in]
        B = lora_state[key_B]  # [m, r] or [out, r]
        dtype = A.dtype
        A_f = A.to(torch.float32).cuda()
        B_f = B.to(torch.float32).cuda()

        # Load source spectral basis and target Cfft map
        U_s = svd_src[base_weight_name]["U"].to(torch.float32).cuda()
        S_s = svd_src[base_weight_name]["S"].to(torch.float32).cuda()
        Vh_s = svd_src[base_weight_name]["Vh"].to(torch.float32).cuda()
        Cfft_layer = dict_Cfft[base_weight_name].to(torch.float32).cuda()
        
        # Apply transfer function
        B_new, A_new = transfer_fn(B_f, A_f, U_s, S_s, Vh_s, Cfft_layer, **kwargs)

        new_state[key_A] = A_new
        new_state[key_B] = B_new
        print(f"[OK] transported LoRA for {base_weight_name}")


    for k, v in list(new_state.items()):
        if isinstance(v, torch.Tensor):
            new_state[k] = v.contiguous()
    
    return new_state
