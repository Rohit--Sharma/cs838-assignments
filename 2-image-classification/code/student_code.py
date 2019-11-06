from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math
from utils import resize_image
import custom_transforms as transforms

#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):

  @staticmethod
  def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
    """
    Forward propagation of convolution operation.
    We only consider square filters with equal stride/padding in width and height!

    Args:
      input_feats: input feature map of size N * C_i * H * W
      weight: filter weight of size C_o * C_i * K * K
      bias: (optional) filter bias of size C_o
      stride: (int, optional) stride for the convolution. Default: 1
      padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

    Outputs:
      output: responses of the convolution  w*x+b

    """
    # sanity check
    assert weight.size(2) == weight.size(3)
    assert input_feats.size(1) == weight.size(1)
    assert isinstance(stride, int) and (stride > 0)
    assert isinstance(padding, int) and (padding >= 0)

    # save the conv params
    kernel_size = weight.size(2)
    ctx.stride = stride
    ctx.padding = padding
    ctx.input_height = input_feats.size(2)
    ctx.input_width = input_feats.size(3)

    # make sure this is a valid convolution
    assert kernel_size <= (input_feats.size(2) + 2 * padding)
    assert kernel_size <= (input_feats.size(3) + 2 * padding)
    
    output_H = int(np.floor((ctx.input_height + (2 * padding) - kernel_size)/stride) + 1) # (H+2P-K)/S + 1    
    output_W = int(np.floor((ctx.input_width + (2 * padding) - kernel_size)/stride) + 1)  # (W+2P-K)/S + 1

    X = unfold(input_feats, kernel_size=kernel_size, padding=padding, stride=stride)    # Unfold input_feat into [(C_i * kernel_size * kernel_size) * #Patches]
    W = weight.view(weight.size(0), - 1).t()                                            # Unfold weight into [C_o * (C_in * kernel_size * kernel_size)]

    Y = torch.matmul(X.transpose(1, 2), W).transpose(1, 2)                              # Multipying X.T and W to compute Y
    for out_ch in range(bias.shape[0]):
      Y[:,out_ch,:] += bias[out_ch]                                                     # Adding bias

    output = fold(Y, output_size=(output_H, output_W), kernel_size=1)                   # Fold Y into [C_o * output_height * output_weight]

    # save for backward (you need to save the unfolded tensor into ctx)
    ctx.save_for_backward(X, weight, bias)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    """
    Backward propagation of convolution operation

    Args:
      grad_output: gradients of the outputs

    Outputs:
      grad_input: gradients of the input features
      grad_weight: gradients of the convolution weight
      grad_bias: gradients of the bias term

    """
    # unpack tensors and initialize the grads
    X_unfolded, weight, bias = ctx.saved_tensors

    # recover the conv params
    kernel_size = weight.size(2)
    stride = ctx.stride
    padding = ctx.padding
    input_height = ctx.input_height
    input_width = ctx.input_width

    grad_input = grad_weight = grad_bias = None

    # gradient w.r.t params
    dY = unfold(grad_output, kernel_size=1)     # Unfold grad_output
    X_T = X_unfolded.transpose(1, 2)            # Transpose x_unfolded
    dW = torch.matmul(dY, X_T)                  # Multiply dY and dW to compute unfolded weight gradients

    dW = dW.sum(dim=0)                          # Add gradient for all the input images in the batch
    grad_weight = dW.view(weight.size())        # Reshape gradient w.r.t weights to correct shape

    # gradient w.r.t input
    W_T = weight.view(weight.shape[0], -1).t()  # Transpose weights
    dX = torch.matmul(W_T, dY)                  # Multiply W_T and dY to compute unfolded input gradients
    # Fold input gradients to correct shape
    grad_input = fold(dX, output_size=(input_height, input_width), kernel_size=kernel_size, padding=padding, stride=stride)

    if bias is not None and ctx.needs_input_grad[2]:
      # compute the gradients w.r.t. bias (if any)
      grad_bias = grad_output.sum((0, 2, 3))

    return grad_input, grad_weight, grad_bias, None, None

custom_conv2d = CustomConv2DFunction.apply

class CustomConv2d(Module):
  """
  The same interface as torch.nn.Conv2D
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
         padding=0, dilation=1, groups=1, bias=True):
    super(CustomConv2d, self).__init__()
    assert isinstance(kernel_size, int), "We only support squared filters"
    assert isinstance(stride, int), "We only support equal stride"
    assert isinstance(padding, int), "We only support equal padding"
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # not used (for compatibility)
    self.dilation = dilation
    self.groups = groups

    # register weight and bias as parameters
    self.weight = nn.Parameter(torch.Tensor(
      out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
  	# initialization using Kaiming uniform
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input):
    # call our custom conv2d op
    return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

  def extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
         ', stride={stride}, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    return s.format(**self.__dict__)

#################################################################################
# Part II: Design and train a network
#################################################################################
class BlockConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, conv_op=nn.Conv2d, **kwargs):
    super(BlockConv2d, self).__init__()
    self.eps = 0.001
    self.block = nn.Sequential(
      conv_op(in_channels, out_channels, **kwargs),
      nn.BatchNorm2d(out_channels, self.eps),
      nn.ReLU(inplace=True))
  def forward(self, x):
    x = self.block(x)
    return x

class BlockA(nn.Module):
  def __init__(self, in_channels, pool_features):
    super(BlockA, self).__init__()
    self.branchA = BlockConv2d(in_channels, 64, kernel_size=1)
    
    self.branchB_1 = BlockConv2d(in_channels, 48, kernel_size=1)
    self.branchB_2 = BlockConv2d(48, 64, kernel_size=5, padding=2)

    self.branchC_1 = BlockConv2d(in_channels, 64, kernel_size=1)
    self.branchC_2 = BlockConv2d(64, 96, kernel_size=3, padding=1)
    self.branchC_3 = BlockConv2d(96, 96, kernel_size=3, padding=1)

    self.branch_pool = BlockConv2d(in_channels, pool_features, kernel_size=1)
    self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    branchA = self.branchA(x)

    branchB_1 = self.branchB_1(x)
    branchB_2 = self.branchB_2(branchB_1)
    
    branchC_1 = self.branchC_1(x)
    branchC_2 = self.branchC_2(branchC_1)
    branchC_3 = self.branchC_3(branchC_2)
    
    branch_pool = self.avg_pool(x)
    branch_pool = self.branch_pool(branch_pool)

    return torch.cat([branchA, branchB_2, branchC_3, branch_pool], 1)

class OurBestNet(nn.Module):
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(OurBestNet, self).__init__()
    # TODO:Add batchnorm
    self.block_conv = BlockConv2d(3, 32, kernel_size=3, stride=2)
    self.block_conv_2 = BlockConv2d(32, 64, kernel_size=3, padding=1)
    self.block_conv_3 = BlockConv2d(64, 120, kernel_size=3)
    self.blockA_1 = BlockA(120, pool_features=32)
    self.blockA_2 = BlockA(256, pool_features=64)
    self.blockA_3 = BlockA(288, pool_features=64)
    self.blockA_4 = BlockA(288, pool_features=64)
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(288, num_classes)

  def reset_parameters(self):
    # init all params
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.consintat_(m.bias, 0.0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)  

  def forward(self, x):
    #print(x.shape)
    x = self.block_conv(x)
    #print("block_conv : " , x.shape)
    x = self.block_conv_2(x)
    #print("block_conv 2 : " , x.shape)
    x = self.block_conv_3(x)
    #print("block_conv 3 : " , x.shape)
    x = self.max_pool(x)
    #print("max_pool : " , x.shape)
    x = self.blockA_1(x)
    #print("blockA_1: " , x.shape)
    x = self.blockA_2(x)
    #print("blockA_2: " , x.shape)
    x = self.blockA_3(x)
    #print("blockA_3: " , x.shape)
    x = self.blockA_4(x)
    #print("blockA_4: " , x.shape)
    x = self.avgpool(x)
    #print("avgpool : " , x.shape)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    #print("fc : ", x.shape)   
    return x

class SimpleNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
    # you can start from here and create a better model
    self.features = nn.Sequential(
      # conv1 block: conv 7x7
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      conv_op(256, 128, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(128, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(128, 512, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

    # Initialise PGD for adversarial training.
    self.attacker = default_attack(nn.CrossEntropyLoss(), num_steps=5)

  def reset_parameters(self):
    # init all params
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.consintat_(m.bias, 0.0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

  def forward(self, x):
    if self.training:
      self.eval()
      # generate adversarial sample based on x
      x = self.attacker.perturb(self, x)
      self.train()
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

# change this to your model!
default_model = SimpleNet
# default_model = OurBestNet  # Use this to train using OurBestNet 

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms(normalize):
  train_transforms = []
  train_transforms.append(transforms.Scale(160))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomColor(0.15))
  train_transforms.append(transforms.RandomRotate(15))
  train_transforms.append(transforms.RandomSizedCrop(128))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  return train_transforms

#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
  def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
    """
    Attack a network by Project Gradient Descent. The attacker performs
    k steps of gradient descent of step size a, while always staying
    within the range of epsilon (under l infinity norm) from the input image.

    Args:
      loss_fn: loss function used for the attack
      num_steps: (int) number of steps for PGD
      step_size: (float) step size of PGD
      epsilon: (float) the range of acceptable samples
               for our normalization, 0.1 ~ 6 pixel levels
    """
    self.loss_fn = loss_fn
    self.num_steps = num_steps
    self.step_size = step_size
    self.epsilon = epsilon

  def perturb(self, model, input):
    """
    Given input image X (torch tensor), return an adversarial sample
    (torch tensor) using PGD of the least confident label.

    See https://openreview.net/pdf?id=rJzIBfZAb

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) an adversarial sample of the given network
    """
    # clone the input tensor and disable the gradients
    output = input.clone()
    input.requires_grad = False
    # enable the gradients on the clone to generate adversarial samples
    output.requires_grad = True

    # Set model in eval mode. This takes care of dropout layers for ex.
    model.eval()

    # loop over the number of steps
    for _ in range(self.num_steps):
      # Forward propagate the input and find the prediction class with least probability
      pred = model(output)
      _, least_conf_pred = torch.min(pred, 1)
      # Using the least confident label as proxy for wrong label to generate adversarial
      #  samples, compute the loss
      loss = self.loss_fn(pred, least_conf_pred)

      # Backward propagate to compute gradients w.r.t the input image
      model.zero_grad()
      loss.backward()

      # Get the direction of the gradient w.r.t input to perturb the image
      output_grad = output.grad.data
      output_grad_sign = torch.sign(output_grad)

      # Fast Gradient sign step to perturb the input
      output.data = output.data + self.step_size * output_grad_sign
      # Clamp the input to epsilon boundary
      output.data = torch.max(torch.min(output.data, input + self.epsilon), input - self.epsilon)
      output.grad.zero_()

    return output

default_attack = PGDAttack


class GradAttention(object):
  def __init__(self, loss_fn):
    """
    Visualize a network's decision using gradients

    Args:
      loss_fn: loss function used for the attack
    """
    self.loss_fn = loss_fn

  def explain(self, model, input):
    """
    Given input image X (torch tensor), return a saliency map
    (torch tensor) by computing the max of abs values of the gradients
    given by the predicted label

    See https://arxiv.org/pdf/1312.6034.pdf

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) a saliency map of size N * 1 * H * W
    """
    # make sure input receive grads
    input.requires_grad = True
    if input.grad is not None:
      input.grad.zero_()

    # Forward propagate to get the actual prediction
    pred = model(input)

    # Here, we want to minimize loss of the most confident prediction. 
    #  Computing actual target with highest prediction and computing its loss
    _, most_conf_pred = torch.max(pred, 1)
    loss = self.loss_fn(pred, most_conf_pred)

    # Backward propagate to compute gradients wrt all requires_grad=True variables (here, also input)
    loss.backward()
    output = input.grad

    # Take the max value of abs(gradient) across each channel to get the saliency map
    output, _ = torch.max(torch.abs(output), 1, keepdim=True)

    return output

default_attention = GradAttention

def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
  """
  Given input image X (torch tensor) and a saliency map
  (torch tensor), compose the visualziations

  Args:
    input: (torch tensor) input image of size N * C * H * W
    output: (torch tensor) input map of size N * 1 * H * W

  Outputs:
    output: (torch tensor) visualizations of size 3 * HH * WW
  """
  # concat all images into a big picture
  input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
  if vis_output is not None:
    output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

    # somewhat awkward in PyTorch
    # add attention to R channel
    mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
    mask = (output_maps[0, :, :] > vis_alpha * output_maps[0,:,:].mean())
    mask = mask.float()
    input_imgs[0,:,:] = torch.max(input_imgs[0,:,:], mask)
  output = input_imgs
  return output

default_visfunction = vis_grad_attention
