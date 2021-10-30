import numpy as np
import torch.nn as nn
import vae_helpers


class VDVAEPyGenBrixLayer(nn.Module):
    def __init__(self, H, pygenbrix_layer):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = vae_helpers.get_conv(H.width, pygenbrix_layer.params_size(H.image_channels), kernel_size=1, stride=1, padding=0)
        self.pygenbrix_layer = pygenbrix_layer
        self.ndims = H.image_channels * H.image_size * H.image_size

    def nll(self, px_z, x):
        xperm = x.permute(0, 3, 1, 2)
        return -self.pygenbrix_layer(self.forward(px_z)).log_prob(xperm)["log_prob"] / self.ndims

    def forward(self, px_z):
        return self.out_conv(px_z)

    def sample(self, px_z):
        im = self.pygenbrix_layer(self.forward(px_z)).sample()
        return im


class VDVAEDiscMixtureLayer(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

    def forward(self, x):
        return VDVAEDiscMixtureDistribution(logits=x, num_mixtures=self.H.num_mixtures)

    def params_size(self, channels):
        return self.H.num_mixtures * 10


#Might not be correct for ffhq_256 
class VDVAEDiscMixtureDistribution():
    def __init__(self, logits, num_mixtures):
        super().__init__()
        self.logits = logits.permute(0, 2, 3, 1)
        self.num_mixtures = num_mixtures

    def log_prob(self, x):
        xperm = x.permute(0, 2, 3, 1)
        ndims = np.prod(x.shape[1:])
        return {"log_prob": -ndims*vae_helpers.discretized_mix_logistic_loss(x=(xperm-.5)*2, l=self.logits)}

    def sample(self):
        return ((vae_helpers.sample_from_discretized_mix_logistic(self.logits, self.num_mixtures)+1.0)*0.5).permute((0,3,1,2))
