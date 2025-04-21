# a new design of spiking including minus spike

from typing import Callable
import torch
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly import visualizing
from matplotlib import pyplot as plt

class SignedIFNode(neuron.IFNode):
    def __init__(self,surrogate_function: surrogate.ATan(),detach_reset: bool = False):
    # def __init__(self, v_threshold: float = 1, v_reset: float = 0, surrogate_function: Callable[..., Any] = ..., detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):
        # super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        super().__init__()
        self.pos_cnt = 0
        self.neg_threshold = torch.tensor(1e-3)
        self.neg_spike = torch.tensor(-1.0)

    # 脉冲发放
    def neuronal_fire(self):
        # 负脉冲
        if self.pos_cnt > 0 and self.v <= self.neg_threshold:
            self.spike = self.neg_spike
        # 负脉冲以外情况
        else:
            self.spike = self.surrogate_function(self.v - self.v_threshold)
            self.pos_cnt += 1
        return self.spike
    
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x + self.surrogate_function(self.pos_cnt - torch.tensor(1)) * self.surrogate_function(self.v - self.neg_threshold)
        return super().neuronal_charge(x)
    
    # 脉冲重置
    def neuronal_reset(self, spike):
        self.pos_cnt = 0
        return super().neuronal_reset(spike)
    
class SignedLIFNode(neuron.LIFNode):
    def __init__(self,surrogate_function: surrogate.ATan(),detach_reset: bool = False):
        super().__init__()
        # self.pos_cnt = 0
        self.alpha = 0.1
        self.neg_threshold = (-1.0 / self.alpha) * self.v_threshold
        self.neg_spike = torch.tensor(-1.0)

    # 脉冲发放
    def neuronal_fire(self):
        # 负脉冲
        if self.v <= self.neg_threshold:
            self.spike = self.neg_spike
        # 负脉冲以外情况
        else:
            self.spike = self.surrogate_function(self.v - self.v_threshold)
            self.pos_cnt += 1
        return self.spike
    
    # 膜电位充电，带负脉冲应该要改
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x + self.surrogate_function(self.pos_cnt - torch.tensor(1)) * self.surrogate_function(self.v - self.neg_threshold)

    
    # 脉冲重置，应该也要改
    def neuronal_reset(self, spike):
        return super().neuronal_reset(spike)

sif_layer = SignedIFNode(surrogate_function=surrogate.ATan())
slif_layer = SignedLIFNode(surrogate_function=surrogate.ATan())

T = 4
N = 1
x_seq = torch.rand([T, N])
print(f'x_seq={x_seq}')

for t in range(T):
    # yt = sif_layer(x_seq[t])
    yt = slif_layer(x_seq[t])
    # print(f'sif_layer.v[{t}]={sif_layer.v}')
    print(f'slif_layer.v[{t}]={slif_layer.v}')

# sif_layer.reset()
slif_layer.reset()
sif_layer.step_mode = 'm'
y_seq = sif_layer(x_seq)
print(f'y_seq={y_seq}')
# sif_layer.reset()
slif_layer.reset()