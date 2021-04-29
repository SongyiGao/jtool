import math
import jittor as jt

from jtool.distributions import Distribution

class Normal(Distribution):
    
    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.log_scale = jt.log(self.scale)

    def log_prob(self, x):
        return -((x - self.loc) ** 2) / (2 * self.variance) - self.log_scale - math.log(math.sqrt(2 * math.pi))

    def sample(self, n):
        shape = (n,) + self.loc.shape
        with jt.no_grad():
            eps = jt.randn(shape)

            return self.loc + self.scale * eps
    
    def entropy(self):
        return self.log_scale + 0.5 * math.log(2 * math.pi * math.e)
    

if __name__ == "__main__":
    import numpy as np
    import torch

    loc = np.random.random([1,3]).astype(np.float32)
    scale = np.random.random([1,3]).astype(np.float32)
    x = np.random.random([1,3]).astype(np.float32)

    dist_jt = Normal(jt.array(loc), jt.array(scale))
    dist_torch = torch.distributions.Normal(torch.tensor(loc), torch.tensor(scale))

    print(dist_jt.loc.numpy(), dist_torch.loc.numpy())
    print(dist_jt.scale.numpy(), dist_torch.scale.numpy())
    print(dist_jt.log_prob(jt.array(x)).numpy(),dist_torch.log_prob(torch.tensor(x)).numpy())
    print(dist_jt.entropy().numpy(),dist_torch.entropy().numpy())