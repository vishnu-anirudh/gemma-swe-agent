"""Parameter-Efficient Fine-Tuning (PEFT) extensions, including Dynamic Rank LoRA."""

from src.peft.dr_lora import DRLoRALinear, DRLoRAManager
from src.peft.saliency import ModuleSaliencyTracker

__all__ = ["DRLoRALinear", "DRLoRAManager", "ModuleSaliencyTracker"]
