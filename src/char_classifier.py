import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from data_utils import (get_train_data, data_path, char_image_size, threshold)
from tensorflow import keras
from sklearn.utils import class_weight
import numpy as np
import modellib
import json
import cv2
from tqdm import tqdm

char_images_path = data_path/'char_images'

def augment(image):
    return [threshold(image)]

def image_generator(data):
    for item in tqdm(data, desc='creating chars .. '):
        iid, shape, labels, path = item
        image = cv2.imread(str(path))
        images = augment(image)
        for image in images:
            yield image, iid, labels

def object_generator(images, size):
    for image, image_id, labels in images:
        for j, label in enumerate(labels):
            c, x, y, w, h = label
            maxwh = max(w, h)
            maxx = min(x+maxwh, image.shape[1])
            maxy = min(y+maxwh, image.shape[0])
            charimg = image[y:maxy, x:maxx, :]
            charimg = cv2.resize(charimg, size)
            yield charimg, image_id, label

def create_char_data(size):
    training_data = get_train_data()
    doc_images =  image_generator(training_data)
    char_images = object_generator(doc_images, size)
    seen_chars = set()

    index = 0
    for charimg, iid, label in tqdm(char_images):
        index += 1
        c, x, y, w, h = label
        directory = char_images_path / c
        if c not in seen_chars:
            os.makedirs(directory)

        seen_chars.add(c)
        fname = directory / '{img}_{j}.jpg'.format(img=str(iid), j=index)
        cv2.imwrite(str(fname), charimg)

    return False

def lr_schedule():
    def lrs(epoch):
        lr = 0.0005
        if epoch >= 5: lr = 0.0001
        if epoch >= 10: lr = 0.00005
        if epoch >= 30: lr = 0.00001
        return lr
    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def get_generators():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.,
                                                                 validation_split=0.1)
    char_indices = data_path / 'char_indices.json'

    args = dict(char_images_path,
        target_size=(char_image_size, char_image_size),
        class_mode='categorical',
        batch_size=128,
        shuffle=True)

    train_generator = train_datagen.flow_from_directory(**args, subset='training')

    json.dump(train_generator.class_indices, open(char_indices, 'w'))

    val_generator = train_datagen.flow_from_directory(**args, subset='validation')

    return train_generator, val_generator

def train_classifier(num_epochs=30):
    imshape = (char_image_size, char_image_size, 3)

    train_generator, val_generator = get_generators()

    class_weights = class_weight.compute_class_weight(
                   'balanced',
                    np.unique(train_generator.classes),
                    train_generator.classes)

    model = modellib.get_char_model(train_generator.num_classes, imshape=imshape, load=False)
    print (model.summary())

    model_path = modellib.get_char_model_path()
    cp = modellib.checkpoint(model_path)
    lrs = lr_schedule()

    history = model.fit_generator(train_generator,
                                  validation_data=val_generator,
                                  epochs=num_epochs,
                                  callbacks=[lrs, cp],
                                  use_multiprocessing=True,
                                  workers=4,
                                  class_weight=class_weights)


    return history

if __name__ == '__main__':
    # create_char_data((char_image_size, char_image_size))
    train_classifier()
