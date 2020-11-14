import cv2
import numpy as np


class RandAugment(object):
    def __init__(self, size=None, number=6, max_magnitude=10):
        self.h, self.w = size
        self.number = number
        self.max_magnitude = max_magnitude
        self.transforms = ["autocontrast", "equalize", "rotate", "solarize", "color", "posterize", "contrast",
                           "brightness", "sharpness", "shearX", "shearY", "translateX", "translateY"]
        self.ranges = {
            "shearX": np.linspace(0, 0.3, self.max_magnitude),
            "shearY": np.linspace(0, 0.3, self.max_magnitude),
            "translateX": np.linspace(0, 0.2, self.max_magnitude),
            "translateY": np.linspace(0, 0.2, self.max_magnitude),
            "rotate": np.linspace(0, 360, self.max_magnitude),
            "color": np.linspace(0.0, 0.9, self.max_magnitude),
            "posterize": np.round(np.linspace(8, 4, self.max_magnitude), 0).astype(np.int),
            "solarize": np.linspace(256, 231, self.max_magnitude),
            "contrast": np.linspace(0.0, 0.5, self.max_magnitude),
            "sharpness": np.linspace(0.0, 0.9, self.max_magnitude),
            "brightness": np.linspace(0.0, 0.3, self.max_magnitude),
            "autocontrast": [0] * self.max_magnitude,
            "equalize": [0] * self.max_magnitude,
            "invert": [0] * self.max_magnitude,
        }
        self.functions = {
            "shearX": self.shearX,
            "shearY": self.shearY,
            "translateX": self.translateX,
            "translateY": self.translateY,
            "rotate": self.rotate,
            "color": self.color,
            "posterize": self.posterize,
            "solarize": self.solarize,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "autocontrast": self.autocontrast,
            "equalize": lambda img, m: img,
            "invert": self.invert,
        }

    def __call__(self, img):
        assert img.shape[0] == self.h and img.shape[1] == self.w

        magnitudes = np.random.randint(0, self.max_magnitude, self.number)
        operations = np.random.choice(self.transforms, self.number)
        for op, mag in zip(operations, magnitudes):
            img = self.functions[op](img, self.ranges[op][mag])
        return img

    def shearX(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        M = np.array([[1, -m, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (self.w, self.h))

    def shearY(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        M = np.array([[1, 0, 0], [-m, 1, 0]])
        return cv2.warpAffine(img, M, (self.w, self.h))

    def translateX(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        M = np.array([[1, 0, -m * self.w], [0, 1, 0]])
        return cv2.warpAffine(img, M, (self.w, self.h))

    def translateY(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        M = np.array([[1, 0, 0], [0, 1, -m * self.h]])
        return cv2.warpAffine(img, M, (self.w, self.h))

    def rotate(self, img, m):
        M = cv2.getRotationMatrix2D((self.w / 2, self.h / 2), m, 1.0)
        return cv2.warpAffine(img, M, (self.w, self.h))

    def color(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        _img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        img = img * (1. + m) - _img * m + 1e-9

        img[img > 255.] = 255.
        img[img < 0.] = 0.
        return img.astype(np.uint8, copy=False)

    def posterize(self, img, m):
        table = np.arange(0, 256, 1)
        mask = ~(2 ** (8 - m) - 1)
        for i in range(256):
            table[i] = i & mask
        return table[img].astype(np.uint8, copy=False)

    def solarize(self, img, m):
        table = np.arange(0, 256, 1)
        for i in range(256):
            if i >= m: table[i] = 255 - i
        return table[img].astype(np.uint8, copy=False)

    def contrast(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()
        img = img * (1. + m) - int(mean + 0.5) * m + 1e-9

        img[img > 255.] = 255.
        img[img < 0.] = 0.
        return img.astype(np.uint8, copy=False)

    def sharpness(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        img = img.astype(np.float, copy=False)
        filter = np.array([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13.
        _img = cv2.filter2D(img, -1, filter)
        for i in [0, -1]:
            _img[i, :, :] = img[i, :, :]
            _img[:, i, :] = img[:, i, :]
        _img += 0.5
        _img = _img.astype(np.uint8, copy=False)
        img = img * (1. + m) - _img * m + 1e-9

        img[img > 255.] = 255.
        img[img < 0.] = 0.
        return img.astype(np.uint8, copy=False)

    def brightness(self, img, m):
        m *= np.random.choice([-1, 1], 1)[0]
        _img = np.ones(m.shape, dtype=np.float)
        img = img * (1. + m) - _img * m + 1e-9

        img[img > 255.] = 255.
        img[img < 0.] = 0.
        return img.astype(np.uint8, copy=False)

    def autocontrast(self, img, m):
        hist = []
        for i in range(3):
            hist += list(cv2.calcHist([img], [i], None, [256], [0, 256]).reshape(-1).astype(int))
        table = []
        for layer in range(0, len(hist), 256):
            h = hist[layer: layer + 256]
            lo, hi = 0, 255
            for i in range(256):
                if h[i]:
                    lo = i
                    break
            for j in range(255, -1, -1):
                if h[j]:
                    hi = j
                    break
            if hi <= lo:
                table += list(range(256))
            else:
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                for ix in range(256):
                    ix = int(ix * scale + offset)
                    if ix < 0:
                        ix = 0
                    elif ix > 255:
                        ix = 255
                    table.append(ix)
        return np.asarray(table)[img].astype(np.uint8)

    def invert(self, img, m):
        return 255 - img
