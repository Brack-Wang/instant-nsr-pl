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
        # whether to add extra output
        self.extra_output = False
        # whether to add extra input
        self.extra_input = True
        # whether to add position as input, note open extra_input
        self.position_input = False
        self.n_frequencies = 8
        self.n_masking_step = 0

        if self.extra_output == True:
            self.n_output_dims = 4
        else:
            self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)

        position_encoding = get_position_encoding(1, self.n_frequencies, self.n_masking_step)

        if self.extra_input == True:
            if self.position_input == True:
                self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims + 2 * self.n_frequencies + 3
                print("extra input position too")
            else:
                self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims + 2 * self.n_frequencies
                print("extra input")
        else:
            self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims 
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    

        self.encoding = encoding
        self.position_encoding = position_encoding
        self.network = network
    
    def forward(self, features, dirs, lightid, positions, *args):

        # print("features:", features.shape)
        # print("dirs:", dirs.shape)
        # print("positions:", positions.shape)
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims)) # N,16
        
        device = dirs.device    
        lights = torch.full((int(dirs.shape[0]), 1), 1 / int(lightid), device=device)
        lights_emd = self.position_encoding(lights.view(-1,1)) # N, 2*self.n_frequencies
        # print("dirs_embd:", dirs_embd.shape)
        # print("lights_emd:", lights_emd.shape)

        if self.extra_input == True:
            if self.position_input == True:
                network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, lights_emd, positions] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
            else:
                network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, lights_emd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        else:
            network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)

        output = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        color = output[:, :3]
        if self.extra_output == True:
            light_id = output[:, 3]
        else:
            light_id = torch.tensor([0], device=device)

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
