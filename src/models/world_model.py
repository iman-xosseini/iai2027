"""Action-free temporal world model: shared encoder + temporal predictor (adapted from LeWorldModel)."""
from torch import nn

from .encoder import Encoder
from .predictor import TemporalPredictor


class TemporalWorldModel(nn.Module):
    """Encodes 24 h of history and the following 6 h with ONE encoder, and predicts the latter's latent.

    No decoder, classification head, action input, stop-gradient or EMA target encoder: the target
    branch uses the same trainable encoder as the history branch. Losses live outside the model (Stage 3).
    """

    def __init__(self, block_size=6, num_sensors=4, context_length=4, latent_dim=32,
                 encoder_hidden_dims=(64, 64), predictor_hidden_dims=(128, 64)):
        super().__init__()
        self.encoder = Encoder(2 * block_size * num_sensors, encoder_hidden_dims, latent_dim)
        self.predictor = TemporalPredictor(context_length, latent_dim, predictor_hidden_dims)

    @classmethod
    def from_config(cls, config):
        """Build the model from the dict loaded from configs/base.yaml."""
        data_cfg, model_cfg = config["data"], config["model"]
        model = cls(
            block_size=data_cfg["block_size"],
            num_sensors=data_cfg["num_sensors"],
            context_length=data_cfg["context_length"],
            latent_dim=model_cfg["latent_dim"],
            encoder_hidden_dims=model_cfg["encoder_hidden_dims"],
            predictor_hidden_dims=model_cfg["predictor_hidden_dims"],
        )
        # The derived input/output sizes must agree with the ones recorded in the config.
        assert model.encoder.input_dim == model_cfg["encoder_input_dim"]
        assert model.encoder.latent_dim == model_cfg["encoder_output_dim"]
        assert model.predictor.net[0].in_features == model_cfg["predictor_input_dim"]
        assert model.predictor.latent_dim == model_cfg["predictor_output_dim"]
        return model

    def encode_history(self, context, context_mask):
        """[B, 4, 6, 4] history and mask -> [B, 4, 32] latents z_{t-3}, ..., z_t."""
        return self.encoder.encode_blocks(context, context_mask)

    def encode_target(self, target, target_mask):
        """[B, 6, 4] future block and mask -> [B, 32] latent z_{t+1}. The target is clean, so its mask is all ones."""
        return self.encoder.encode_blocks(target, target_mask)

    def forward(self, context, context_mask, target, target_mask):
        z_history = self.encode_history(context, context_mask)  # [B, 4, 32]
        z_pred = self.predictor(z_history)                      # [B, 32], from the history only
        z_target = self.encode_target(target, target_mask)      # [B, 32], same encoder parameters
        return {"z_history": z_history, "z_pred": z_pred, "z_target": z_target}
