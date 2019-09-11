from pathlib import Path
import csv
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras as keras
import json
from skimage.morphology import skeletonize

data_path = Path('../data/').expanduser()

train_images_path = data_path/'train_images'
test_images_path = data_path/'test_images'
train_csv_path = data_path/'train.csv'
test_csv_path = data_path/'sample_submission.csv'

submission_path = data_path/'submission.csv'
cache_path = data_path / 'cache'

n_masks = 3
n_chars = 4196
char_image_size = 96
pool_size_heat = 2
pool_size_wh = 2

char_indices = json.load(open(data_path / 'char_indices.json'))
index2char = {char_indices[char]:char for char in char_indices}

def parse_labels(labels):
    label_parts = labels.split()
    res = []
    for r1, r2 in zip(range(0, len(label_parts), 5), range(5, len(label_parts), 5)):
        codepoint, x, y, w, h = label_parts[r1:r2]
        x,y,w,h = map(int, [x,y,w,h])
        res.append((codepoint, x,y,w,h))
    return res

def get_train_data():
    return get_label_data(train_csv_path, train_images_path)

def get_test_data():
    data = []
    reader = csv.reader(open(test_csv_path))
    next(reader)
    for image_id, label in tqdm(reader, desc='loading labels ..'):
        labels = []
        path = (test_images_path / image_id).with_suffix('.jpg')
        image_size = Image.open(path).size
        image_size = image_size[1], image_size[0]
        data.append((image_id, image_size, labels, path))
    return data

def get_label_data(csv_path, images_path):
    data = []
    reader = csv.reader(open(csv_path))
    next(reader)
    for image_id, label in tqdm(reader, desc='loading labels ..'):
        labels = parse_labels(label)
        path = (images_path/image_id).with_suffix('.jpg')
        image_size = Image.open(path).size
        image_size = image_size[1], image_size[0]
        data.append((image_id, image_size, labels, path))
    return data

def threshold(image, thinning=False, soft=False):
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    ret,im = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im = 255 - im
    iters = 2 if soft else 4
    kernel = np.ones((2,2),np.uint8)
    im = cv2.dilate(im,kernel,iterations = iters)
    im = cv2.erode(im,kernel,iterations = iters)
    if thinning:
        im = skeletonize(im/255)
        im = np.asarray(im*255, dtype=np.uint8)
        im = cv2.dilate(im,kernel,iterations = 2)

    im = cv2.cvtColor(im, code=cv2.COLOR_GRAY2BGR)
    return im

def create_center_mask(shape, labels, img_size):
    # centerpoint, w_ratio, h_ratio
    masks = np.zeros((img_size, img_size, n_masks), dtype=np.float16)
    for label in labels:
        c,x,y,w,h = label
        x_ratio = img_size / shape[1]
        y_ratio = img_size / shape[0]

        # Set center point to 1 surrounded by a gaussian
        x, w = int(x*x_ratio), int(w*x_ratio)
        y, h = int(y*y_ratio), int(h*y_ratio)
        h_ratio = h / img_size
        w_ratio = w / img_size
        h, w = masks[y:y + h, x:x + w, 0].shape
        gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        d = np.sqrt(gx * gx + gy * gy)
        sigma, mu = 0.2, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        masks[y:y+h, x:x+w, 0] = g
        masks[y+h//2, x+w//2, 0] = 1

        # create another channel and set
        masks[y + h // 2, x + w // 2, 1] = h_ratio
        masks[y + h // 2, x + w // 2, 2] = w_ratio

    return masks

def get_image(path, size=None):
    img = cv2.imread(str(path))
    if size:
        img = cv2.resize(img, size)
    return img

def filter_by_indices(indices, lists):
    filtered_lists = []
    for l in lists:
        lnew = []
        for i in indices:
            lnew.append(l[i])
        filtered_lists.append(lnew)

    return filtered_lists

def filter_by_threshold(threshold, scores, lists):
    indices = []
    for i,s in enumerate(scores):
        if s > threshold:
            indices.append(i)

    return filter_by_indices(indices, lists)

def rescale_coords(boxes, s1, s2):
    w_ratio = s1[0] / s2[0]
    h_ratio = s1[1] / s2[1]
    rescaled = []
    for x,y,w,h in boxes:
        x, y, w, h = x * w_ratio, y * h_ratio, w * s1[0], h * s1[1]
        rescaled.append([x,y,w,h])
    return rescaled

def draw_boxes(image, boxes):
    image = image.copy()
    rectimage = image.copy()
    cv2.rectangle(rectimage, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), -1)
    for x, y, w, h in boxes:
        bbx1 = max(int(x - w // 2), 0)
        bbx2 = int(x + w // 2)
        bby1 = max(int(y - h // 2), 0)
        bby2 = int(y + h // 2)
        cv2.rectangle(rectimage, (bby1, bbx1), (bby2, bbx2), (255, 255, 0), -1)
    image = cv2.addWeighted(rectimage, 0.5, image, 1 - 0.5, 0)
    return image

def get_chars_from_image(image, boxes):
    char_images = []
    for x, y, w, h in boxes:
        maxwh = max(w, h)
        bbx1 = max(int(x - maxwh // 2), 0)
        bbx2 = int(x + maxwh // 2)
        bby1 =  max(int(y - maxwh // 2), 0)
        bby2 =  int(y + maxwh // 2)
        char_image = image[bbx1:bbx2, bby1:bby2]
        char_image = cv2.resize(char_image, (char_image_size, char_image_size))
        char_image = char_image[...,::-1] / 255.
        char_images.append(char_image)
    return char_images

class Display:
    def __init__(self, maxsize, time):
        self.maxsize = maxsize
        self.time = time
        self.mode = True
        self.writers = {}

    def getimage(self, image):
        size_ratio = self.maxsize / max(image.shape[0], image.shape[1])
        new_size = int(image.shape[1] * size_ratio), int(image.shape[0] * size_ratio)
        return cv2.resize(image, new_size)

    def show(self, image, name, time=None):
        if not self.mode:
            return

        time = time if time is not None else self.time
        cv2.imshow(name, self.getimage(image))
        cv2.waitKey(time)
        return 0

    def save(self, image, name):
        image = self.getimage(image)
        if name not in self.writers:
            writer = cv2.VideoWriter(name,
                                     cv2.VideoWriter_fourcc('M','J','P','G'),
                                     30, (image.shape[1], image.shape[0]))
            self.writers[name] = writer

        self.writers[name].write(image)

    def off(self):
        self.mode = False

    def on(self):
        self.mode = True

    def show_sample_result(self, image, model, img_size, box_threshold, num_boxes):
        original_shape = image.shape

        self.show(image, "Original image")
        image_mask_pred = cv2.resize(image, (img_size, img_size))
        image_mask_pred = np.expand_dims(image_mask_pred, axis=0)
        pred_masks = model.predict(image_mask_pred, batch_size=8)
        # boxes, box_scores = get_boxes(pred_masks[0:1], num_boxes, box_threshold)
        # boxes = rescale_coords(boxes, original_shape, (img_size, img_size))
        # box_image = draw_boxes(image, boxes)
        pred_mask = np.asarray(pred_masks[0][:,:,0] * 255, dtype=np.uint8)
        self.show(pred_mask, "raw boxes")
        # self.save(box_image, "box_evolution.avi")

def nms(intermediate):
    sigmoid = keras.backend.sigmoid(intermediate)
    maxlayer = keras.layers.MaxPool2D((pool_size_heat, pool_size_heat),strides=(1,1), padding='same')(sigmoid)
    keep = keras.backend.cast(keras.backend.equal(maxlayer, sigmoid), keras.backend.floatx())
    res = intermediate * keep
    return res, keep

def get_boxes(pred_masks, top_k=100, threshold=0.0):
    nsamples, width = pred_masks.shape[0], pred_masks.shape[1]

    heat_mask = pred_masks[...,0:1]
    w_mask = pred_masks[...,1:2]
    h_mask = pred_masks[...,2:3]
    heat_mask, keep = nms(heat_mask)
    w_mask = keras.layers.MaxPool2D((pool_size_wh, pool_size_wh),strides=(1,1), padding='same')(w_mask)
    w_mask = w_mask * keep
    h_mask = keras.layers.MaxPool2D((pool_size_wh, pool_size_wh),strides=(1,1), padding='same')(h_mask)
    h_mask = h_mask * keep

    heat_mask_flat = tf.reshape(heat_mask, [-1])
    w_mask_flat = tf.reshape(w_mask, [-1])
    h_mask_flat = tf.reshape(h_mask, [-1])
    values, indices = tf.math.top_k(heat_mask_flat,k=top_k,sorted=True)
    x_indices = tf.cast(indices / width, tf.float32)
    y_indices = tf.cast(indices % width, tf.float32)
    w_values = tf.gather(w_mask_flat, indices)
    h_values = tf.gather(h_mask_flat, indices)
    w_values = tf.cast(w_values, tf.float32)
    h_values = tf.cast(h_values, tf.float32)
    stack = tf.transpose(tf.stack([x_indices, y_indices, w_values, h_values, values]))
    boxes = stack.numpy()
    boxes = list(filter(lambda x:x[4] > threshold, boxes))
    scores = [box[4] for box in boxes]
    boxes = [box[:4] for box in boxes]
    return boxes, scores

def generate_mask_from_boxes(img_size, boxes):
    blank_image = np.zeros((img_size), dtype=np.uint8) + 0.3
    for x,y,w,h in boxes:
        w =  w * img_size[0]
        h = h * img_size[1]
        x,w,y,h = list(map(int, [x,w,y,h]))
        blank_image[x - w // 2:x + w // 2, y - h // 2:y + h // 2] = 1
    return blank_image
