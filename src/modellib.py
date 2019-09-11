import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow import keras
from tensorflow.keras import backend as K
from data_utils import n_chars

def unet_model(output_channels, size):
    # base_model = tf.keras.applications.MobileNetV2(input_shape=[size, size, 3], include_top=False, alpha=.35)
    base_model = extract_backbone_from_char_model(n_chars, imshape=[size, size, 3])

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = True

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        pix2pix.upsample(32, 3),  # 32x32 -> 64x64
    ]

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='sigmoid')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[size, size, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def checkpoint(path):
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            verbose=1)
    return cp

def get_segmentation_model_path(size):
    return '../weights/unet_with_wh_{size}.hdf5'.format(size=size)

def bce_dice_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return tf.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true[..., 0], 1), y_pred[..., 0], tf.ones_like(y_pred[..., 0]))
        pt_0 = tf.where(tf.equal(y_true[..., 0], 0), y_pred[..., 0], tf.zeros_like(y_pred[..., 0]))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        loss =  -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        mask = K.sign(y_true[..., 0] - 1 +  K.epsilon())
        mask = K.clip(mask, 0, 1)
        N = K.sum(mask) + 1

        return loss/N

    return binary_focal_loss_fixed

def all_loss(y_true, y_pred):
    all_loss = (binary_focal_loss(gamma=2., alpha=.25)(y_true, y_pred) + 5.0*size_loss(y_true, y_pred))
    return all_loss


def size_loss(y_true, y_pred):
    mask = K.sign(y_true[..., 0] - 1 + K.epsilon())
    mask = K.clip(mask, 0, 1)
    N = K.sum(mask) + 1
    sizeloss = K.sum(K.abs(y_true[...,1]-y_pred[...,1]*mask)+K.abs(y_true[...,2]-y_pred[...,2]*mask))
    return sizeloss/N

def get_char_model_path():
    return "../weights/char_classifier.hdf5"

def get_char_model(n_classes, imshape, load=True):
    model_path = get_char_model_path()
    base_model = keras.applications.MobileNetV2(input_shape=imshape,
                                            include_top=False,
                                            alpha=0.75,
                                            pooling='avg')

    model = keras.Sequential([
        base_model,
        keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    if load:
        status = model.load_weights(model_path, by_name=False)
        print (status)

    return model

def extract_backbone_from_char_model(n_classes, imshape):
    base_model = get_char_model(n_classes, imshape)
    backbone_with_pooling = base_model.layers[0]
    backbone_layers_without_pooling = backbone_with_pooling.layers[:-1]
    model = keras.models.Model(inputs=backbone_layers_without_pooling[0].input,
                               outputs=backbone_layers_without_pooling[-1].output)
    return model

