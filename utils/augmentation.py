import numpy as np
from PIL import Image

class RandomFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:, ::-1]
            label = label[:, ::-1]
        return image, label

class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            W, H, C = image.shape
            h1 = np.random.randint(0, H*self.crop_rate)
            w1 = np.random.randint(0, W*self.crop_rate)
            h2 = np.random.randint(H*(1-self.crop_rate), H+1)
            w2 = np.random.randint(W*(1-self.crop_rate), W+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]
            
        return image, label

class RandomCropOut():
    def __init__(self, crop_rate=0.2, prob=1.0):
        self.crop_rate = crop_rate
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            W, H, C = image.shape

            h1 = np.random.randint(0, H*self.crop_rate)
            w1 = np.random.randint(0, W*self.crop_rate)
            h2 = int(h1 + H * self.crop_rate)
            w2 = int(w1 + W * self.crop_rate)

            image[w1:w2, h1:h2] = 0
            label[w1:w2, h1:h2] = 0

        return image, label

class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            bright_factor = np.random.uniform(1-self.bright_range, 1+self.bright_range)
            image = (image * bright_factor).astype(image.dtype)

        return image, label

class RandomNoise():
    def __init__(self, noise_range=5, prob=0.9):
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            W, H, C = image.shape
            noise = np.random.randint(-self.noise_range, self.noise_range, (W, H, C))
            image = (image + noise).clip(0, 255).astype(image.dtype)
        
        return image, label