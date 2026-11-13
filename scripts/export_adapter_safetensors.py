#!/usr/bin/env python3
"""Export PyTorch DR-LoRA Checkpoints to Standard HuggingFace PEFT safetensors Format.

Converts local checkpoints (e.g. checkpoints/sft_dr_lora/adapter_model.pt) into:
- adapter_model.safetensors
- adapter_config.json
Ready for Google ADK / Kaggle Gemma Challenge submission packaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from rich.console import Console
from safetensors.torch import save_file

console = Console()


def export_adapter(
    checkpoint_path: str,
    output_dir: str,
    base_model_name: str = "google/gemma-4-31b-it-qat-w4a16-ct",
    lora_alpha: int = 16,
) -> None:
    ckpt_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    console.print(f"[cyan]Loading checkpoint from {ckpt_path}...[/cyan]")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights: Dict[str, Dict[str, torch.Tensor]] = ckpt.get("adapter_weights", {})
    if not weights:
        raise ValueError("No 'adapter_weights' found in checkpoint dictionary.")

    tensors: Dict[str, torch.Tensor] = {}
    target_modules = set()
    max_rank = 64

    for mod_name, mod_dict in weights.items():
        # Infer module suffix (e.g. q_proj, v_proj)
        suffix = mod_name.split(".")[-1]
        target_modules.add(suffix)

        lora_a = mod_dict["lora_A"].contiguous()
        lora_b = mod_dict["lora_B"].contiguous()
        max_rank = max(max_rank, lora_a.shape[0])

        # Standard HuggingFace PEFT parameter naming convention:
        tensors[f"base_model.model.{mod_name}.lora_A.weight"] = lora_a
        tensors[f"base_model.model.{mod_name}.lora_B.weight"] = lora_b

    safetensors_path = out_dir / "adapter_model.safetensors"
    save_file(tensors, str(safetensors_path))
    console.print(f"[green]Saved {len(tensors)} tensors to {safetensors_path} ({safetensors_path.stat().st_size / 1e6:.2f} MB)[/green]")

    # Write PEFT adapter_config.json
    config_payload = {
        "base_model_name_or_path": base_model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": max_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(list(target_modules)),
        "modules_to_save": None,
    }

    config_path = out_dir / "adapter_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    console.print(f"[green]Saved PEFT config to {config_path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DR-LoRA Checkpoint to PEFT safetensors")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/sft_dr_lora/adapter_model.pt",
        help="Path to input checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./submission/adapters/main_lora",
        help="Directory to save adapter_model.safetensors and adapter_config.json",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-4-31b-it-qat-w4a16-ct",
        help="Target base model identifier",
    )
    args = parser.parse_args()

    export_adapter(args.checkpoint, args.output_dir, base_model_name=args.base_model)


if __name__ == "__main__":
    main()
