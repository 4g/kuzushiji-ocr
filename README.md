**Kaggle's Kuzushiji Recognition**

###### Keras Solution for 73% accuracy

* Mobilenet backend => small, super fast (also, works on mobiles)
* U-net => pix2pix upsampling with mobilenet backend
* Centernet with max-pooling vs NMS (speed vs accuracy, more training)
* Thresholded images => (6% improvement)
* Patches of original images for training
* Imprinting to alleviate generalization loss
* Cyclic learning rates
* Display Callbacks in Keras

![Masks gif](resources/masks.gif)