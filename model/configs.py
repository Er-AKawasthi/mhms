# configs.py
# Cleaned and simplified for running ViT-B/16 on local datasets with no ResNet hybrid.
import ml_collections
from ml_collections import ConfigDict

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration for standard usage."""
    config = ConfigDict()
    config.patches = ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_testing():
    """Returns a minimal configuration for testing only."""
    config = ConfigDict()
    config.patches = ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'testing': get_testing(),
}
