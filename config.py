from addict import Dict
from modules.utils import get_norm_layer


class TrainerConfig:
    """
    Configuration class for the trainer.
    """

    batch_size = 128
    lambda_cycle = 10
    lambda_iden = 5  
    beta1 = 0.5  
    beta2 = 0.999
    num_epoch = 160
    lr = 2e-4  
    
    path_data = 'AAPM_data'
    path_checkpoint = 'CT_denoising_best'
    
    supervised: bool = False

class ModelConfig:

    G_F2Q: Dict = Dict({
        'args': (),
        'kwargs': {
            'input_nc': 1,
            'output_nc': 1,
            'ngf': 64,
            'norm_layer': get_norm_layer('batch'),
            'use_dropout': False
        }
    })
    
    G_Q2F: Dict = Dict({
        'args': (),
        'kwargs': {
            'input_nc': 1,
            'output_nc': 1,
            'ngf': 64,
            'norm_layer': get_norm_layer('batch'),
            'use_dropout': False
        }
    })
    
    G_Q2F_sup: Dict = Dict({
        'args': (),
        'kwargs': {
            'input_nc': 1,
            'output_nc': 1,
            'ngf': 64,
            'norm_layer': get_norm_layer('batch'),
            'use_dropout': False
        }
    })
    
    D_F: Dict = Dict({
        'args': (),
        'kwargs': {
            'in_channels': 1,
            'ndf': 64,
            'num_layers': 3,
            'normalization_layer': get_norm_layer('batch')
        }
    })
    
    D_Q: Dict = Dict({
        'args': (),
        'kwargs': {
            'in_channels': 1,
            'ndf': 64,
            'num_layers': 3,
            'normalization_layer': get_norm_layer('batch')
        }
    })
    
    model_name: str = 'supervised_best'