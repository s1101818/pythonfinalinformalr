import torch
import torch.nn as nn
from .util import init, get_clones
import argparse
from typing import List, Tuple, Union, Optional

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
    ):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        if torch.isinf(x).any(): #inf 檢測play輸入
            print("\nmlplayer input x inf: ",torch.isinf(x).sum().item())
            print("infpos:\n",torch.isinf(x))
            print(x)
            #input("pause now")
        x2=x
        x = self.fc1(x)
        if torch.isnan(x).any(): #nan 在liner連接處理後檢測
            print("\nmlplayer after fc1 x nan: ",torch.isnan(x).sum().item())
            print("origin x ",x2)
            print("max:",torch.max(x2))
            print("\nnan ver x",x)
            #input("pause now")
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        if torch.isnan(x).any():
            print("\nmlplayer output x nan: ",torch.isnan(x).sum().item())
            #input("pause now")
        return x


class MLPBase(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        obs_shape: Union[List, Tuple],
        override_obs_dim: Optional[int] = None,
    ):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        # override_obs_dim is only used for graph-based models
        if override_obs_dim is None:
            obs_dim = obs_shape[0]
        else:
            print("Overriding Observation dimension")
            obs_dim = override_obs_dim

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
        )

    def forward(self, x: torch.tensor):
        if torch.isinf(x).any(): #inf nomaliz前檢查
            print("\nmlp input inf:",torch.isinf(x).sum().item())
            print("mlp input infpos:\n",torch.isinf(x))
            print(x)
            print("max:",torch.max(x))
            #input("pause now")
        if self._use_feature_normalization:
            #print("x is in normallize")
            x = self.feature_norm(x)
        if torch.isinf(x).any(): #inf nomaliz後檢查
            print("\nmlp input inf:",torch.isinf(x).sum().item())
            print(x)
            print("max:",torch.max(x))
            #input("pause now")
        
        x = self.mlp(x)
        if torch.isnan(x).any(): #nan mlp處理後檢查
            print("\nmlp forward output x nan: ",torch.isnan(x).sum().item())
            #input("pause now")
        return x