import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import modellib
import cv2
import numpy as np
from tqdm import tqdm
import json
from nms import nms
import math

from data_utils import (get_train_data, get_test_data,
                        n_masks, data_path, Display, get_image,
                        filter_by_indices, filter_by_threshold,
                        rescale_coords, draw_boxes, get_chars_from_image,
                        index2char, threshold, get_boxes)


img_size = (512, 512)
char_image_size = 96
num_boxes = 10000
box_threshold = 0.4
n_chars = 4196
out_fname = 'test_submission.csv'


mask_cnn = modellib.unet_model(n_masks, img_size[0])
mask_cnn.load_weights(modellib.get_segmentation_model_path(img_size[0]))
char_cnn = modellib.get_char_model(n_chars, imshape=(char_image_size, char_image_size, 3))
display = Display(800, 100)
display.off()

def get_val_data():
    data = get_train_data()
    import random
    random.seed(42)
    random.shuffle(data)
    val_percent = 0.1
    train_size = int(len(data) * (1 - val_percent))
    training_data, val_data = data[0:train_size], data[train_size:]
    data = val_data
    return data

def join_images(images):
    n_images = len(images)
    w = int(math.sqrt(n_images))
    h = n_images // w
    s2 = None
    for i in range(h):
        s1 = np.hstack(images[i*w : i*w + w])
        if s2 is None:
            s2 = s1
            continue
        s2 = np.vstack((s1, s2))
    return s2

def get_predictions(image):
    original_shape = image.shape
    image_mask_pred = cv2.resize(image, img_size)
    display.show(image_mask_pred, "image")
    image_mask_pred = np.expand_dims(image_mask_pred, axis=0) / 255.
    pred_masks = mask_cnn.predict(image_mask_pred, batch_size=8)
    boxes, box_scores = get_boxes(pred_masks[0:1], num_boxes, box_threshold)
    boxes = rescale_coords(boxes, original_shape, img_size)
    box_image = draw_boxes(image, boxes)
    display.show(box_image, "boxes")

    char_images = get_chars_from_image(image, boxes)
    char_prediction_scores = []
    if len(boxes) != 0:
        char_prediction_scores = char_cnn.predict(np.asarray(char_images))

    #display.show(join_images(char_images), "chars")

    cindices = [np.argmax(score) for score in char_prediction_scores]
    char_scores = [score[i] for i, score in zip(cindices, char_prediction_scores)]
    chars = [index2char[cindex] for cindex in cindices]
    lists = [chars, boxes, char_scores, char_images]
    chars, boxes, char_scores, char_images = filter_by_threshold(10.0/n_chars, char_scores, lists)
    nms_indices = nms.boxes(boxes, char_scores)
    chars, boxes, char_scores, char_images = filter_by_indices(nms_indices, lists)
    display.show(box_image, "missing boxes")
    return chars, boxes, char_scores, char_images

def eval_prediction(chars, pred_boxes, char_scores, labels):
    point_in_box = lambda cx,cy,x,y,w,h: x < cx < x + w and y < cy < y + h
    truth_covered = set()
    pred_covered = set()
    fn = 0

    for truth_index in range(len(labels)):
        c, y, x, h, w = labels[truth_index]
        for i in range(len(chars)):
            pred_char, b = chars[i], pred_boxes[i]
            cx, cy, _, _ = b
            if point_in_box(cx,cy,x,y,w,h) and c == pred_char:
                truth_covered.add(truth_index)
                pred_covered.add(i)
                break

    tp = len(truth_covered)
    fp = len(chars) - len(pred_covered)
    fn = len(labels) - len(truth_covered)
    return tp, fp, fn

def compute_f1(tp, fp, fn):
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision > 0 and recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1, precision, recall

def eval():
    data = get_val_data()
    tp, fp, fn = 0, 0, 0
    total_chars_predicted = 0
    total_chars_truth = 0
    for sample_index in tqdm(range(len(data))):
        image_id, size, labels, path = data[sample_index]
        image = get_image(path)
        image = threshold(image)
        chars, boxes, char_scores, char_images = get_predictions(image)
        total_chars_predicted += len(chars)
        total_chars_truth += len(labels)
        _tp, _fp, _fn = eval_prediction(chars, boxes, char_scores, labels)
        tp += _tp
        fp += _fp
        fn += _fn

    f1, precision, recall =  compute_f1(tp, fp, fn)
    print (num_boxes, box_threshold, char_image_size, total_chars_predicted, total_chars_truth)
    print ("tp:{tp} fp:{fp} fn:{fn} precision:{p} recall:{r} f1:{f1}".format(tp=tp, fp=fp, fn=fn, p=precision, r=recall, f1=f1))


def test():
    output_file = data_path / out_fname
    output = open(output_file, 'w')
    output.write("image_id,labels\n")

    data = get_test_data()
    for sample_index in tqdm(range(len(data))):
        image_id, size, labels, path = data[sample_index]
        image = get_image(path)
        image = threshold(image)
        chars, boxes, char_scores, char_images = get_predictions(image)
        prep_string = ""
        for char, box in zip(chars, boxes):
            y,x,_,_ = box
            prep_string += "{c} {x} {y} ".format(c=char, x=x, y=y)

        prep_string = image_id + "," + prep_string + "\n"
        output.write(prep_string)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'eval':
        eval()
    if sys.argv[1] == 'test':
        test()
