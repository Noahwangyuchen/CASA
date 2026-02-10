import torch
import utils
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def identity_transfer(B, A, U_s, S_s, Vh_s, Cdst):
    return B, A


def cluster_aware_spectral_arbitration(B, A, 
                                  U_s, S_s, Vh_s, 
                                  Cfft,  
                                  rotation_threshold=0.5,
                                  q_threshold=0.5,
                                  arbitrate_q=0.85,
                                  target_rank=32,
                                  lora_scale=1.0):
    """
    核心逻辑：
    1. 密度分级: 确定 High Mask (保护区) 和 Low Mask (背景区)。
    2. 混合评分指标 (Hybrid Metric):
       - Score = Pixel_Energy * Context_Factor
       - Pixel_Energy = |Cs * Cdst| (点积能量)
       - Context_Factor:
         >> Top-K 区域内: Block Cosine Similarity (簇级方向一致性)
         >> Top-K 区域外: 1.0 
    3. 策略:
       - 在 Low Mask 区域直接置为 Cs - Cdst，力求还原 LoRA 作用。
       - 在 High Mask 区域内的同号像素中，计算 Score 的分位数。
       - Score > Threshold 的像素 -> 仲裁成 max(Cs, Cdst) - Cdst 防止过度激活。
       - 其他 -> 置为 Cs。
    """
    print(f"--- Starting CASA ---")
    
    # =========================================================
    # Step 0: 基础定义与投影 (保持不变)
    # =========================================================
    device = B.device
    D_out, D_in = Cfft.shape

    B = B * lora_scale
    A = A * lora_scale
    print(f"Scalings: B={lora_scale}, A={lora_scale}")
    
    M_left = U_s.t() @ B
    M_right = A @ Vh_s.t()
    Cs = M_left @ M_right

    if Cs.shape[0] == Cs.shape[1]: 
        energies = S_s ** 2
        cumulative_energy = torch.cumsum(energies, dim=0) / energies.sum()
        k_dynamic = torch.where(cumulative_energy >= 0.90)[0]
        k = k_dynamic[0].item() + 1 if len(k_dynamic) > 0 else S_s.size(0)
    else:
        k = S_s.size(0)
    
    # =========================================================
    # Step 1: 微扰分簇 (保持不变)
    # =========================================================
    Cs_k = Cs[:k, :k]
    S_k = S_s[:k]
    
    M_force = torch.abs(Cs_k)
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
    # Step 2: 密度筛选 (单阈值 q_threshold)
    # =========================================================
    Cdst_k = Cfft[:k, :k]
    M_mat = torch.nn.functional.one_hot(labels.long(), num_classes=n_clusters).float().t()
    Cluster_Sizes = M_mat.sum(dim=1) + 1e-6
    
    Rx_Density = (M_mat @ torch.norm(Cdst_k, dim=1)) / Cluster_Sizes
    Tx_Density = (M_mat @ torch.norm(Cdst_k, dim=0)) / Cluster_Sizes
    
    rx_th = torch.quantile(Rx_Density, q_threshold)
    tx_th = torch.quantile(Tx_Density, q_threshold)
    
    cl_rx_high = torch.where(Rx_Density >= rx_th)[0]
    cl_tx_high = torch.where(Tx_Density >= tx_th)[0]
    
    # 构造 High Mask (覆盖全量)
    idx_rx = torch.isin(labels, cl_rx_high)
    idx_tx = torch.isin(labels, cl_tx_high)
    full_rows = torch.zeros(D_out, dtype=torch.bool, device=device); full_rows[:k] = idx_rx
    full_cols = torch.zeros(D_in, dtype=torch.bool, device=device); full_cols[:k] = idx_tx
    mask_high = full_rows.unsqueeze(1) | full_cols.unsqueeze(0)

    # =========================================================
    # Step 3: 构建混合评分矩阵 (Hybrid Score Matrix)
    # =========================================================
    
    # 1. 计算 Pixel 级能量
    pixel_energy_map = (Cs * Cfft).abs()
    
    # 2. 计算 Block 级 Cosine (仅在 Top-K 内)
    dot_prod = M_mat @ (Cs_k * Cdst_k) @ M_mat.t()
    norm_cs = torch.sqrt(M_mat @ (Cs_k**2) @ M_mat.t())
    norm_cdst = torch.sqrt(M_mat @ (Cdst_k**2) @ M_mat.t())
    block_cosine = dot_prod / (norm_cs * norm_cdst + 1e-8)
    
    # 3. 将 Block Cosine 映射回 Pixel 空间 (Context Factor)
    # 初始化全为 1.0 (Top-K 以外的区域默认乘 1)
    context_factor_map = torch.ones((D_out, D_in), device=device, dtype=Cs.dtype)
    
    # 在 Top-K 区域填入对应的 Block Cosine
    # M_mat.T [k, n] @ [n, n] @ [n, k] -> [k, k]
    pixel_level_block_cosine = M_mat.t() @ block_cosine @ M_mat
    context_factor_map[:k, :k] = pixel_level_block_cosine
    
    # 4. 最终混合得分
    # Score = Pixel_Energy * Context_Factor
    # 注意: 如果 block cosine 是负的(方向冲突)，这里得分为负，自然会被 quantile 淘汰
    hybrid_score = pixel_energy_map * context_factor_map

    # =========================================================
    # Step 4: 策略执行
    # =========================================================
    mask_diff = (Cs.sign() != Cfft.sign())
    mask_same = (Cs.sign() == Cfft.sign())
    
    Cs_new = Cs - Cfft
    
    Cs_new[mask_high] = Cs[mask_high]
    # 2. High & 同号 -> 基于 Hybrid Score 仲裁
    mask_high_same = mask_high & mask_same
    
    # 提取待筛选区域的分数
    if mask_high_same.sum() > 0:
        candidate_scores = hybrid_score[mask_high_same]
        
        # 计算阈值
        num_elements = candidate_scores.numel()
        max_samples = 10_000_000
        if num_elements > max_samples:
            # 随机采样
            indices = torch.randperm(num_elements, device=candidate_scores.device)[:max_samples]
            sample_for_quantile = candidate_scores[indices]
            score_thresh = torch.quantile(sample_for_quantile, arbitrate_q)
        else:
            score_thresh = torch.quantile(candidate_scores, arbitrate_q)
        
        mask_suppress = mask_high_same & (hybrid_score > score_thresh)

        max_val = torch.where(Cs.abs() > Cfft.abs(), Cs, Cfft)
        C_arbitrated = max_val - Cfft
        Cs_new[mask_suppress] = C_arbitrated[mask_suppress]
        print(f"Suppressed {mask_suppress.sum().item()} pixels.")

    # =========================================================
    # Step 4: 重构
    # =========================================================
    U_c, S_c, V_c = torch.svd_lowrank(Cs_new, q=target_rank, niter=6)
    Vh_c = V_c.t() 
    
    sqrt_S = torch.diag(torch.sqrt(S_c))
    B_new = U_s @ (U_c @ sqrt_S)
    A_new = (sqrt_S @ Vh_c) @ Vh_s
    
    return B_new, A_new


def transfer_lora(
    lora_state,
    svd_src,
    dict_Cdst,
    transfer_method,
    ignore_keywords=None,
    transfer_kwargs=None,
):
    """
    lora_state: LoRA 参数字典
    svd_src, svd_tgt: dict[layer_name] -> {"U":..., "S":..., "Vh":...}
    output_path: 保存 spectral transport 后的 LoRA
    transfer_method: 指定 transfer 方法，可以是函数或预设的名称
    transfer_kwargs: 透传给具体 transfer 方法的参数
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

    # 透传参数（默认兼容 use_basis_transfer）
    kwargs = dict(transfer_kwargs or {})

    for key in list(lora_state.keys()):
        if key in processed or (ignore_keywords and any(kw in key for kw in ignore_keywords)):
            continue
        
        # 识别 LoRA A 的 key
        if key.endswith("lora_A.weight"):
            prefix = key[:-len(".lora_A.weight")]
            key_A = key
            key_B = prefix + ".lora_B.weight"
        elif key.endswith("lora_down.weight"):
            prefix = key[:-len(".lora_down.weight")]
            key_A = key
            key_B = prefix + ".lora_up.weight"
        else:
            # 非 A/down 的权重先不动，暂存一下，后面统一拷贝
            continue

        print(f"Transfering LoRA as {prefix} ...")

        if key_B not in lora_state:
            # 没找到成对的 B/up，直接原样拷贝 A 再说
            print(f"[WARN] Pair not found for {key_A}, skip spectral transport.")
            new_state[key_A] = lora_state[key_A]
            continue

        processed.add(key_A)
        processed.add(key_B)

        # 推出 base 权重的层名，比如 blocks.0.self_attn.q.weight
        svd_keys = list(svd_src.keys())  # or svd_wan_1_3.keys()
        base_weight_name = utils.auto_map_lora_key_to_svd_key(prefix, svd_keys)

        if base_weight_name not in svd_src:
            # print(f"[WARN] {base_weight_name} not found in SVD dicts, copy LoRA as-is.")
            # new_state[key_A] = lora_state[key_A]
            # new_state[key_B] = lora_state[key_B]
            print(f"[WARN] {base_weight_name} not found in SVD dicts, skip it.")
            continue

        # 取出 LoRA A/B
        A = lora_state[key_A]  # [r, n] or [r, in]
        B = lora_state[key_B]  # [m, r] or [out, r]
        dtype = A.dtype
        A_f = A.to(torch.float32).cuda()
        B_f = B.to(torch.float32).cuda()

        # 取源/目标 SVD
        U_s = svd_src[base_weight_name]["U"].to(torch.float32).cuda()
        S_s = svd_src[base_weight_name]["S"].to(torch.float32).cuda()
        Vh_s = svd_src[base_weight_name]["Vh"].to(torch.float32).cuda()
        C_dst = dict_Cdst[base_weight_name].to(torch.float32).cuda()
        
        # 调用 transfer 函数
        B_new, A_new = transfer_fn(B_f, A_f, U_s, S_s, Vh_s, C_dst, **kwargs)

        new_state[key_A] = A_new
        new_state[key_B] = B_new
        print(f"[OK] transported LoRA for {base_weight_name}")

    # # 把未处理的参数原样拷贝（比如 alpha, scaling, 其他 bias）
    # for key, value in lora_state.items():
    #     if key not in new_state:
    #         new_state[key] = value

    for k, v in list(new_state.items()):
        if isinstance(v, torch.Tensor):
            new_state[k] = v.contiguous()
    
    return new_state

