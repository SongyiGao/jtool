import math
import numpy as np
import jittor as jt

from jtool.distributions import Distribution

def simple_presum(x):
    src = '''
__inline_static__
@python.jittor.auto_parallel(1)
void kernel(int n0, int i0, in0_type* x, in0_type* out, int nl) {
    out[i0*(nl+1)] = 0;
    for (int i=0; i<nl; i++)
        out[i0*(nl+1)+i+1] = out[i0*(nl+1)+i] + x[i0*(nl+1)+i];
}
kernel(in0->num/in0->shape[in0->shape.size()-1], 0, in0_p, out0_p, in0->num);
    '''
    return jt.code(x.shape[:-1]+(x.shape[-1]+1,), x.dtype, [x],
        cpu_src=src, cuda_src=src)

class Categorical(Distribution):
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
    
    def sample(self, sample_shape=[]):
        shape = sample_shape + self.probs.shape[:-1] + (1,)
        rand = jt.rand(shape)
        one_hot = jt.logical_and(self.cum_probs_l < rand, rand <= self.cum_probs_r)
        index = one_hot.index(one_hot.ndim-1)
        return (one_hot * index).sum(-1)
    
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