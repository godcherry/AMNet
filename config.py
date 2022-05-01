from pathlib import Path

__all__ = ['project_path', 'dataset_config']

project_path = Path(__file__).parent


dataset_config = {
    'elliptic':
        {
            'K': 2,
            'M': 4,
            'hidden_channels': 128,
            'lr_f': 5e-2,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'beta': .5,
            'epochs': 2000,
            'patience': 200
        },
    'yelp':
        {
            'K': 2,
            'M': 5,
            'hidden_channels': 64,
            'lr_f': 5e-2,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 2000,
            'patience': 200
        },
    'FinV':
        {
            'K': 2,
            'M': 7,
            'hidden_channels': 128,
            'lr_f': 5e-2,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'beta': 1.,
            'epochs': 2000,
            'patience': 200
        },
    'Telecom':
        {
            'K': 2,
            'M': 6,
            'hidden_channels': 128,
            'lr_f': 5e-2,
            'lr': 5e-4,
            'weight_decay': 5e-4,
            'beta': 1.,
            'epochs': 2000,
            'patience': 200
        }

}