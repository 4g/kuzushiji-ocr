import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from data_utils import (get_train_data, get_test_data,
                        Display, n_chars, n_masks, get_image, threshold, create_center_mask, data_path)

import modellib
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import cv2
import numpy as np


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, image, img_size, box_threshold, num_boxes, display):
        self.model = model
        self.image = image
        self.img_size = img_size
        self.box_threshold = box_threshold
        self.num_boxes = num_boxes
        self.display = display

    def on_batch_end(self, epoch, logs=None):
        self.display.show_sample_result(self.image, self.model, self.img_size, self.box_threshold, self.num_boxes)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size, img_size, shuffle=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.data = data
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgb = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        maskb = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        for index, item in enumerate(batch):
            iid, shape, labels, path = item
            img = self.get_image(path)
            mask = create_center_mask(shape, labels, self.img_size)
            imgb[index] = img
            maskb[index] = mask

        return imgb, maskb

    def get_image(self, path):
        img = cv2.imread(str(path))
        img = threshold(img)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.
        return img

    def sample(self):
        iid, shape, labels, path = self.data[0]
        return self.get_image(path)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.data)

def lr_schedule():
    def lrs(epoch):
        lr = 0.0002
        if epoch >= 5: lr = 0.0001
        if epoch >= 15: lr = 0.00005
        if epoch >= 30: lr = 0.00001
        return lr
    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def train():
    img_size = 512
    num_boxes = 500
    box_threshold = 0.2
    char_image_size = 96

    model = modellib.unet_model(n_masks, img_size)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=modellib.all_loss,
                  metrics=[modellib.binary_focal_loss(gamma=2., alpha=.25), modellib.size_loss])


    data = get_train_data()
    val_split = 0.1
    train_size = int(len(data)*(1 - val_split))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_generator = DataGenerator(train_data, 8, img_size)
    val_generator = DataGenerator(val_data, 8, img_size)

    display = Display(600, 10)
    display.on()
    # display_callback = DisplayCallback(model, val_generator.sample(), img_size, box_threshold, num_boxes, display)

    model_path = modellib.get_segmentation_model_path(img_size)
    model.load_weights(modellib.get_segmentation_model_path(img_size))

    cp = modellib.checkpoint(model_path)
    callbacks = [lr_schedule(), cp]

    history = model.fit(train_generator,
                        validation_data = val_generator,
                          epochs=50,
                          shuffle=True,
                          callbacks=callbacks,
                          use_multiprocessing=True,
                          verbose=1,
                          workers=4,)

train()