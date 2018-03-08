import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.Tensor(out_features))
    self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
    self.register_buffer('bias_epsilon', torch.Tensor(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.weight_mu.size(1))
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

  def _scale_noise(self, size):
    x = torch.randn(size)
    x = x.sign().mul(x.abs().sqrt())
    return x

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(self._scale_noise(self.out_features))

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon)), self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon)))
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc_h = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, args.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * args.atoms, std_init=args.noisy_std)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.fc_h(x.view(-1, 3136)))
    v, a = self.fc_z_v(x), self.fc_z_a(x)  # Calculate value and advantage streams
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    p = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return p.clamp(min=1e-12, max=1 - 1e-12)  # Use clipping to prevent NaNs

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
