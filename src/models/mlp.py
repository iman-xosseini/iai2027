from torch import nn


def build_mlp(input_dim, hidden_dims, output_dim):
    """Linear -> GELU for every hidden width, then a plain Linear output layer.

    The output layer has no activation or normalization: the latent space stays unconstrained
    (no softmax, sigmoid or unit-norm), which is what the SIGReg objective of Stage 3 expects.
    """
    layers = []
    width = input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.GELU()]
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)
