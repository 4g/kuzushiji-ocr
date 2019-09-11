from tensorflow import keras
import numpy as np

class WrappedGenerator(keras.callbacks.Callback):
    def __init__(self, generator):
        self.generator = generator
        self.set()

    def set(self, mixup=0.2, increment_size=0.05):
        self.mixup = mixup
        self.increment_size = increment_size

    def stream(self):
        prev_x, prev_y = [], []
        for x, y in self.generator:
            if len(prev_x) > 0:
                mix_x = (1.0 - self.mixup) * x  + self.mixup * prev_x
                mix_y = (1.0 - self.mixup) * y  + self.mixup * prev_y
                yield np.concatenate((x, mix_x), axis=0), np.concatenate((y, mix_y), axis=0)

            prev_x = x
            prev_y = y


    def __len__(self):
        return len(self.generator)