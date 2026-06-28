"""C-MAE augmented MAG model for EyeBench."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model import ConditionalGazeMAE, ConditionalGazeMAEConfig
from src.configs.constants import TaskTypes
from src.configs.data import DataArgs
from src.configs.models.dl.CMAEMAG import CMAEMAGEye
from src.configs.trainers import TrainerDL
from src.models.base_model import register_model
from src.models.mag_model import MAGModel


@register_model
class CMAEMAGModel(MAGModel):
    """MAG classifier with a text-conditioned masked gaze encoder."""

    def __init__(
        self,
        model_args: CMAEMAGEye,
        trainer_args: TrainerDL,
        data_args: DataArgs,
    ) -> None:
        super().__init__(
            model_args=model_args,
            trainer_args=trainer_args,
            data_args=data_args,
        )
        self.cmae_args = model_args
        reconstruction_dim = model_args.cmae_reconstruction_dim
        if reconstruction_dim is None:
            reconstruction_dim = min(len(model_args.eye_features), model_args.eyes_dim)

        self.cmae = ConditionalGazeMAE(
            ConditionalGazeMAEConfig(
                gaze_dim=model_args.eyes_dim,
                text_dim=model_args.text_dim,
                reconstruction_dim=reconstruction_dim,
                hidden_dim=model_args.cmae_hidden_dim,
                num_layers=model_args.cmae_num_layers,
                num_heads=model_args.cmae_num_heads,
                ff_dim=model_args.cmae_ff_dim,
                dropout=model_args.cmae_dropout,
                mask_ratio=model_args.cmae_mask_ratio,
            )
        )
        self.cmae_classifier = nn.Sequential(
            nn.Dropout(model_args.cmae_classifier_dropout),
            nn.Linear(model_args.cmae_hidden_dim, self.num_classes),
        )
        self.cmae_logit_gate = nn.Parameter(
            torch.tensor(float(model_args.cmae_logit_gate_init))
        )

    @staticmethod
    def _align_modal_sequences(
        gaze_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = min(
            gaze_features.size(1),
            text_features.size(1),
            attention_mask.size(1),
        )
        return (
            gaze_features[:, :seq_len, :],
            text_features[:, :seq_len, :],
            attention_mask[:, :seq_len],
        )

    def _select_text_features(self, output: Any) -> torch.Tensor:
        hidden_states = getattr(output, 'hidden_states', None)
        if not hidden_states:
            raise ValueError('CMAEMAGModel requires output_hidden_states=True')

        layer_index = self.cmae_args.cmae_condition_layer
        if layer_index < 0:
            layer_index = len(hidden_states) + layer_index
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise ValueError(
                f'cmae_condition_layer={self.cmae_args.cmae_condition_layer} '
                f'is outside hidden_states length {len(hidden_states)}'
            )
        return hidden_states[layer_index]

    def shared_step(
        self,
        batch: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_data = self.unpack_batch(batch)
        eye_token_type_ids = None

        assert batch_data.input_masks is not None, 'input_masks cannot be None'
        assert batch_data.grouped_inversions is not None, (
            'grouped_inversions cannot be None'
        )

        gaze_features, _, attention_mask = self.finalize_eye_data(batch_data)
        assert gaze_features is not None, 'CMAEMAGModel requires gaze features'

        labels = batch_data.labels
        if self.task == TaskTypes.REGRESSION:
            labels = labels.squeeze().float()

        output = self(
            input_ids=batch_data.input_ids,
            attention_mask=attention_mask,
            labels=None,
            gaze_features=gaze_features,
            gaze_positions=batch_data.grouped_inversions,
            output_hidden_states=True,
            eye_token_type_ids=eye_token_type_ids,
        )

        mag_logits = output.logits
        text_features = self._select_text_features(output)
        cmae_gaze, cmae_text, cmae_mask = self._align_modal_sequences(
            gaze_features=gaze_features,
            text_features=text_features,
            attention_mask=attention_mask,
        )
        mask_ratio = self.cmae_args.cmae_mask_ratio if self.training else 0.0
        cmae_output = self.cmae(
            gaze_features=cmae_gaze,
            text_features=cmae_text,
            attention_mask=cmae_mask,
            mask_ratio=mask_ratio,
        )

        cmae_logits = self.cmae_classifier(cmae_output.trial_representation)
        gate = torch.sigmoid(self.cmae_logit_gate)
        logits = mag_logits + gate * cmae_logits

        if self.task == TaskTypes.REGRESSION:
            logits = logits.squeeze()
        if logits.ndim == 1 and self.task != TaskTypes.REGRESSION:
            logits = logits.unsqueeze(0)

        classification_loss = self.loss(logits, labels)
        use_aux_loss = (
            self.training
            or self.cmae_args.cmae_apply_reconstruction_to_eval_loss
        )
        if use_aux_loss:
            auxiliary_loss = (
                self.cmae_args.cmae_reconstruction_loss_weight
                * cmae_output.reconstruction_loss
                + self.cmae_args.cmae_alignment_loss_weight
                * cmae_output.alignment_loss
            )
        else:
            auxiliary_loss = classification_loss.new_tensor(0.0)
        loss = classification_loss + auxiliary_loss

        return labels, loss, logits
