import numpy as np

class MixUp(object):
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, images, labels):
        batch_size = images.shape[0]
        indices = np.random.permutation(batch_size)
        lam = np.random.beta(self.alpha, self.beta)
        images = images * lam + images[indices] * (1. - lam)
        labels = labels * lam + labels[indices] * (1. - lam)
        return images, labels
