import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from patchify import patchify
from config import cf
def load_dataset(path,path1,path2 ):
    """ Loading the images and masks """
    train_x = sorted(glob(os.path.join(path, "images", "*.jpg")))
    valid_x=sorted(glob(os.path.join(path2, "images", "*.jpg")))
    test_x=sorted(glob(os.path.join(path1, "images", "*.jpg")))
    
    train_y= sorted(glob(os.path.join(path, "mask", "*.bmp")))
    valid_y=sorted(glob(os.path.join(path2, "mask", "*.png")))
    test_y=sorted(glob(os.path.join(path1, "mask", "*.bmp")))
    
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image = image / 255.0

    """ Processing to patches """
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)

    return patches

def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_NEAREST)

    # Get binary cup mask
    mask = ((mask == 255) | (mask == 128)).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape(cf["flat_patches_shape"])
    y.set_shape([cf["image_size"], cf["image_size"], 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds