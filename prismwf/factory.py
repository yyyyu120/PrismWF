"""Model construction helpers for the paper and controlled ablations."""

from .model import PrismWF


def ablation_options(ablation: str) -> dict[str, object]:
    if ablation == "no-router":
        return {"enable_router_interaction": False}
    if ablation == "no-router-no-cross":
        return {
            "enable_router_interaction": False,
            "enable_cross_granularity": False,
        }
    if ablation == "single-granularity":
        return {
            "branch_kernels": (5,),
            "enable_router_interaction": False,
            "enable_cross_granularity": False,
        }
    if ablation != "full":
        raise ValueError(f"Unknown ablation: {ablation}")
    return {}


def build_prismwf(
    num_classes: int,
    num_layers: int = 3,
    ablation: str = "full",
) -> PrismWF:
    return PrismWF(
        num_classes=num_classes,
        num_layers=num_layers,
        **ablation_options(ablation),
    )
