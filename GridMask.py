import cv2, random, math
import numpy as np


class GridMask(object):
    def __init__(self, d1=96, d2=224, rotate=1, ratio=0.6, mode=1, prob=0.8):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, image):
        if np.random.rand() > self.prob:
            return image
        h, w, _ = image.shape
        mask = self.mask(h, w).reshape((h, w, 1)).astype(image.dtype)
        return image * mask

    def mask(self, h, w):
        diagonal = math.ceil(math.sqrt(h * h + w * w))
        d = random.randint(self.d1, self.d2 - 1)
        r = int(d * self.ratio)
        x = random.randint(0, d - 1)
        y = random.randint(0, d - 1)
        mask = np.ones((diagonal, diagonal), dtype=np.float)
        _x = x - d + r
        if _x > 0:
            mask[:_x, :] *= 0
        for i in range(x, diagonal, d):
            mask[i:i + r, :] *= 0
        _y = y - d + r
        if _y > 0:
            mask[:, :_y] *= 0
        for i in range(y, diagonal, d):
            mask[:, i:i + r] *= 0

        M = cv2.getRotationMatrix2D((diagonal / 2, diagonal / 2), self.rotate, 1.0)
        mask = cv2.warpAffine(mask, M, (diagonal, diagonal))
        _x = (diagonal - h) // 2
        _y = (diagonal - w) // 2
        mask = mask[_x: _x + h, _y: _y + w]
        if self.mode == 1:
            mask = 1. - mask
        return mask
