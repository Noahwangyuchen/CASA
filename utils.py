import torch
import re
import io
import pickle

# custom unpickler to load SVD results to specified device
class CPU_Unpickler(pickle.Unpickler):
    def __init__(self, file, device):
        super().__init__(file)
        self.device = device

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        else: return super().find_class(module, name)


def load_results(path, device=None):
    if device is None:
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return CPU_Unpickler(f, device).load()


# ==== Layer-name normalization ====
def unify_layer_names_with_rules(svd_dict, mapping_rules, ignore_keywords=None, debug=False):
    if ignore_keywords is None:
        ignore_keywords = []

    new_dict = {}
    for name, value in svd_dict.items():
        # Skip ignored layers
        if any(kw in name for kw in ignore_keywords):
            continue

        new_name = name
        # Apply mapping rules sequentially
        for pattern, replacement in mapping_rules.items():
            if debug:
                print(new_name)
            new_name = re.sub(pattern, replacement, new_name)

        new_dict[new_name] = value
    return new_dict


def auto_map_lora_key_to_svd_key(lora_key, svd_keys, debug=False):
    """
    Input:
      lora_key: e.g. "blocks.0.self_attn.q.lora_A.weight"
      svd_keys: list of all keys in the SVD dict

    Output:
      best_match: e.g. "blocks.0.self_attn.q.weight"
    """

    # 0. Remove LoRA prefix
    base = lora_key.replace("diffusion_model.", "")

    # 1. Remove LoRA suffixes
    base = (
        base
        .replace(".lora_A.weight", "")
        .replace(".lora_B.weight", "")
        .replace(".lora_up.weight", "")
        .replace(".lora_down.weight", "")
    )

    # 2. Common attention alias replacements
    alias_attn = {
        "to_q": "q", "q_proj": "q", "query": "q",
        "to_k": "k", "k_proj": "k", "key": "k",
        "to_v": "v", "v_proj": "v", "value": "v",
        "to_out.0": "o", "out_proj": "o",
    }
    for a, b in alias_attn.items():
        base = base.replace(f".{a}", f".{b}")

    # 3. FFN alias replacements
    alias_ffn = {
        "fc_in": "0",
        "fc1": "0",
        "net.0.proj": "0",
        "mlp.0": "0",
        "fc_out": "2",
        "fc2": "2",
        "net.2": "2",
        "mlp.2": "2",
    }
    for a, b in alias_ffn.items():
        base = base.replace(f".ffn.{a}", f".ffn.{b}")

    # 4. Append ".weight" if missing
    if not base.endswith(".weight"):
        base = base + ".weight"

    return base
