import torch
import numpy as np
import functools
from torch import nn
from torch.nn import init

import matplotlib.pyplot as plt

class Mean:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def __call__(self, value):
        self.sum += value
        self.count += 1

    def result(self):
        return self.sum / self.count


# initialize parameters of neural networks
def init_weights(net):
  def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      init.normal_(m.weight.data, 0.0, 0.02)
      if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, 0.02)
      init.constant_(m.bias.data, 0.0)
    
  print('Initialize network.')
  net.apply(init_func)


# Calculate average loss during one epoch
class Mean:
  def __init__(self):
    self.numel = 0
    self.mean = 0
  
  def __call__(self, val):
    self.mean = self.mean * (self.numel / (self.numel + 1)) + val / (self.numel + 1)
    self.numel += 1
  
  def result(self):
    return self.mean


# Show input and output images during training
def show_imgs(imgs):
  FQF = np.concatenate(imgs[:3], axis=2)
  QFQ = np.concatenate(imgs[3:], axis=2)
  img_array = np.squeeze(np.concatenate([FQF, QFQ], axis=1))

  img_array = img_array * 4000
  img_array = np.clip(img_array, -1000, 1000)

  plt.imshow(img_array, cmap='gray')
  plt.show()
  
# Show input and output images during supervised training
def show_imgs_supervised(imgs):
    concatenated_img = np.concatenate(imgs, axis=2)
    concatenated_img = concatenated_img * 4000
    concatenated_img = np.clip(concatenated_img, -1000, 1000)
    concatenated_img = np.squeeze(concatenated_img)
    plt.imshow(concatenated_img, cmap='gray')
    plt.show()

# Set 'requires_grad' of the networks
def set_requires_grad(nets, requires_grad=False):
  if not isinstance(nets, list):
    nets = [nets]
  for net in nets:
    if net is not None:
      for param in net.parameters():
        param.requires_grad = requires_grad

# Get normalization layer
def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer