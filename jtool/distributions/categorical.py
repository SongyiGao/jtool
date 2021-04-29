import math
import numpy as np
import jittor as jt

from jtool.distributions import Distribution

class Categorical:
    def __init__(self, probs=None, logits=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        self._probs = probs
        self._logits = logits
        
    @property
    def probs(self):
        if self._probs is None:
            return jt.softmax(self.logits, dim=-1)
        else:
            return self._probs
    
    @property
    def logits(self):
        if self._logits is None:
            return jt.log(jt.clamp(self.probs, min_v=eps, max_v=1-eps))
        else:
            return self._logits
        
    @property
    def batch_shape(self):
        return self.probs.shape[:-1]

    def log_prob(self, x):
        return jt.log(self.probs)[0,x]
    
    def sample(self, n):
        shape = (n,) + self.loc.shape
        with jt.no_grad():
            eps = jt.randn(shape)

            return self.loc + self.scale * eps
    
    def entropy(self):
        return -jt.sum(jt.mean(self.probs) * jt.log(self.probs))
    
    
if __name__ == "__main__":
    import numpy as np
    import torch

    p = jt.nn.softmax(jt.array(np.random.random([1,3]))).numpy()
    x = np.array([1]).astype(np.int32)

    dist_jt = Categorical(jt.array(p))
    dist_torch = torch.distributions.Categorical(torch.tensor(p))

    print(dist_jt.log_prob(jt.array(x)))
    print(dist_torch.log_prob(torch.tensor(x)))