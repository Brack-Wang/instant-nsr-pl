import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp, get_position_encoding
from systems.utils import update_module_step


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 4
        # self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)

        self.n_frequencies = 3
        self.n_masking_step = 0
        position_encoding = get_position_encoding(1, self.n_frequencies, self.n_masking_step)

        # self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims + 2 * self.n_frequencies
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims 
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    

        self.encoding = encoding
        self.position_encoding = position_encoding
        self.network = network
    
    def forward(self, features, dirs, lightid, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        # N,16
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        
        device = dirs.device  # Get the device from dirs tensor
        lights = torch.full((int(dirs.shape[0]), 1), 1 / int(lightid), device=device)
        # N, 2*self.n_frequencies
        lights_emd = self.position_encoding(lights.view(-1,1))

        # network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, lights_emd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        # print("network_inp:", network_inp.shape)

        output = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        color = output[:, :3]
        light_id = output[:, 3]
        # light_id = lights
        # print("color:", color.shape)
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color, light_id

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
