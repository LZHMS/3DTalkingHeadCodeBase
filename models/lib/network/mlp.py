import functools
import torch
import torch.nn as nn

# from ..head.build import HEAD_REGISTRY


class MLP(nn.Module):

    def __init__(
        self,
        in_features=2048,
        hidden_layers=[],
        out_features=None,
        activation="relu",
        bn=True,
        dropout=0.0,
        
    ):
        super().__init__()
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        assert len(hidden_layers) > 0
        
        # If out_features is not specified, use the last hidden layer dimension
        if out_features is None:
            out_features = hidden_layers[-1]
        self.out_features = out_features
        self.in_features = in_features

        mlp = []

        if activation == "relu":
            act_fn = functools.partial(nn.ReLU, inplace=True)
        elif activation == "leaky_relu":
            act_fn = functools.partial(nn.LeakyReLU, inplace=True)
        else:
            raise NotImplementedError

        for hidden_dim in hidden_layers:
            mlp += [nn.Linear(in_features, hidden_dim)]
            if bn:
                mlp += [nn.LayerNorm(hidden_dim)]
            mlp += [act_fn()]
            if dropout > 0:
                mlp += [nn.Dropout(dropout)]
            in_features = hidden_dim

        # Add final projection layer if output dimension differs from last hidden layer
        if out_features != hidden_layers[-1]:
            mlp += [nn.Linear(hidden_layers[-1], out_features)]

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # Flatten input if it has more than 2 dimensions
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.mlp(x)


# @HEAD_REGISTRY.register()
# def mlp(**kwargs):
#     return MLP(**kwargs)