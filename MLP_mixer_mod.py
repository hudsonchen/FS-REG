from typing import Any

import einops
# import flax.linen as nn
import haiku as hk
import jax.numpy as jnp
import jax


class MlpBlock:
    def __init__(self,
                 mlp_dim: int,
                 name):
        self.mlp_dim = mlp_dim
        self.name = name

    def __call__(self, x):
        y = hk.Linear(self.mlp_dim)(x)
        y = jax.nn.gelu(y)
        return hk.Linear(x.shape[-1])(y)


class MixerBlock:
    def __init__(self,
                 tokens_mlp_dim: int,
                 channels_mlp_dim: int):
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

    def __call__(self, x):
        y = hk.LayerNorm(axis=-1,
                         create_scale=True,
                         create_offset=True)(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = hk.LayerNorm(axis=-1,
                         create_scale=True,
                         create_offset=True)(x)
        return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)


class MlpMixer:
    def __init__(self,
                 patches: int,
                 num_classes: int,
                 num_blocks: int,
                 hidden_dim: int,
                 tokens_mlp_dim: int,
                 channels_mlp_dim: int):
        self.patches = patches
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

    def __call__(self, inputs, is_training):
        x = hk.Conv2D(self.hidden_dim, self.patches,
                      stride=self.patches, name='stem')(inputs)
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = hk.LayerNorm(axis=-1,
                         create_scale=True,
                         create_offset=True,
                         name='pre_head_layer_norm')(x)
        x = jnp.mean(x, axis=1)
        return hk.Linear(self.num_classes,
                         w_init=jnp.zeros,
                         name='head')(x)
