import torch
import functools
import torch.nn as nn

class ResnetBlock(nn.Module):
    """ Resnet block for constructing a conv block with skip connections. """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """ Builds a convolutional block with optional dropout. """
        conv_block = []
        if padding_type in ['reflect', 'replicate']:
            conv_block += [getattr(nn, f"{padding_type.capitalize()}Pad2d")(1)]
        else:
            p = 1 if padding_type == 'zero' else raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """ Implements skip connections. """
        return x + self.conv_block(x)
    
class ResnetGenerator(nn.Module):
    """ Generator model based on the ResNet architecture. """
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d if not isinstance(norm_layer, functools.partial) else norm_layer.func == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # Add downsampling layers
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # Add ResNet blocks
        mult = 2 ** 2
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)]

        # Add upsampling layers
        for i in range(2, 0, -1):
            mult = 2 ** i
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)
        self.residual_conv = nn.Conv2d(input_nc, output_nc, kernel_size=1, padding=0, bias=True)

    def forward(self, input):
        """ Combines model output with residual. """
        output = self.model(input)
        residual = self.residual_conv(input)
        return nn.Tanh()(output + residual)
    
class Discriminator(nn.Module):
    """ PatchGAN discriminator model for distinguishing real vs generated patches. """
    
    def __init__(self, in_channels, ndf=64, num_layers=3, normalization_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        use_bias = isinstance(normalization_layer, functools.partial) and normalization_layer.func == nn.InstanceNorm2d

        modules = [nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        nf_mult, nf_mult_prev = 1, 1
        
        for n in range(1, num_layers):
            nf_mult_prev, nf_mult = nf_mult, min(8, 2 ** n)
            modules += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                normalization_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        modules += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*modules)

    def forward(self, input_tensor):
        """ Applies the discriminator model to input. """
        return self.model(input_tensor)
