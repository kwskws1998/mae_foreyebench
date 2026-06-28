from dataclasses import dataclass

from src.configs.constants import DLModelNames
from src.configs.models.dl.MAG import MAG
from src.configs.utils import register_model_config


@register_model_config
@dataclass
class CMAEMAGEye(MAG):
    """MAG baseline with text-conditioned masked gaze autoencoding."""

    base_model_name: DLModelNames = DLModelNames.CMAE_MAG_MODEL

    mag_dropout: float = 0.1
    mag_injection_index: int = 23
    cmae_hidden_dim: int = 256
    cmae_num_layers: int = 2
    cmae_num_heads: int = 4
    cmae_ff_dim: int = 512
    cmae_dropout: float = 0.1
    cmae_mask_ratio: float = 0.3
    cmae_reconstruction_loss_weight: float = 0.1
    cmae_alignment_loss_weight: float = 0.0
    cmae_classifier_dropout: float = 0.1
    cmae_logit_gate_init: float = -2.0
    cmae_condition_layer: int = 0
    cmae_apply_reconstruction_to_eval_loss: bool = False
    cmae_reconstruction_dim: int | None = None
