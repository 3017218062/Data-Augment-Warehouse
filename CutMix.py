import random, math
import numpy as np

class CutMix(object):
    def __init__(self, alpha=1., beta=1., constraint=(0., 1.)):
        self.alpha = alpha
        self.beta = beta
        self.constraint = constraint

    def __call__(self, images, labels):
        batch_size, h, w = images.shape[:-1]
        indices = np.random.permutation(batch_size)
        lam = np.random.beta(self.alpha, self.beta)
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = self.box(lam, h, w)
        images[:, bbox_x1:bbox_x2, bbox_y1:bbox_y2, :] = images[indices, bbox_x1:bbox_x2, bbox_y1:bbox_y2, :]
        lam = 1. - (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) / (h * w)
        labels = labels * lam + labels[indices] * (1. - lam)
        return images, labels

    def box(self, lam, h, w):
        rate = math.sqrt(1. - lam)
        cut_h, cut_w = int(h * rate / 2), int(w * rate / 2)
        if self.constraint != (0., 1.):
            cut_h = np.clip(cut_h, int(h * self.constraint[0]), int(h * self.constraint[1]))
            cut_w = np.clip(cut_w, int(w * self.constraint[0]), int(w * self.constraint[1]))
        cut_x, cut_y = random.randint(0, h), random.randint(0, w)
        bbox_x1 = np.clip(cut_x - cut_h, 0, h)
        bbox_y1 = np.clip(cut_y - cut_w, 0, w)
        bbox_x2 = np.clip(cut_x + cut_h, 0, h)
        bbox_y2 = np.clip(cut_y + cut_w, 0, w)
        return bbox_x1, bbox_y1, bbox_x2, bbox_y2
