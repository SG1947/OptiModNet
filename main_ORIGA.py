import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers as L
import os
# from keras_flops import get_flops
from math import log2
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger

def channel_attention(input_tensor, reduction_ratio=16):
    # Global Average Pooling
    channel_avg = L.GlobalAveragePooling2D()(input_tensor)
    channel_avg = L.Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu')(channel_avg)
    channel_avg = L.Dense(input_tensor.shape[-1], activation='sigmoid')(channel_avg)

    # Reshape to original tensor shape
    channel_avg = L.Reshape((1, 1, input_tensor.shape[-1]))(channel_avg)
    return input_tensor * channel_avg


def mlp(x, cf):
    x = L.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    x = L.Dense(cf["hidden_dim"])(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    return x
def transformer_encoder(x, cf):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.GroupQueryAttention(head_dim=cf["head_dim"],
    num_query_heads=cf["num_query_heads"],
    num_key_value_heads=cf["num_key_value_heads"])( x, x)
    x = L.Add()([x, skip_1])

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, cf)
    x = L.Add()([x, skip_2])

    return x

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters,  [3, 3], padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x

def deconv_block(x, num_filters, strides=2):
    x = L.Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=strides)(x)
    return x

def build_unetr_2d(cf):
    """ Inputs """
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs = L.Input(input_shape) ## (None, 256, 3072)

    """ Patch + Position Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs) ## (None, 256, 768)

    positions = tf.range(start=0, limit=cf["num_patches"], delta=1) ## (256,)
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions) ## (256, 768)
    x = patch_embed + pos_embed ## (None, 256, 768)

    """ Transformer Encoder """
    skip_connection_index = [3, 6, 9, 12]
    skip_connections = []

    for i in range(1, cf["num_layers"]+1, 1):
        x = transformer_encoder(x, cf)

        if i in skip_connection_index:
            skip_connections.append(x)

    """ CNN Decoder """
    z3, z6, z9, z12 = skip_connections

    ## Reshaping
    z0 = L.Reshape((cf["image_size"], cf["image_size"], cf["num_channels"]))(inputs)

    shape = (
        cf["image_size"]//cf["patch_size"],
        cf["image_size"]//cf["patch_size"],
        cf["hidden_dim"]
    )
    z3 = L.Reshape(shape)(z3)
    z6 = L.Reshape(shape)(z6)
    z9 = L.Reshape(shape)(z9)
    z12 = L.Reshape(shape)(z12)

    ## Additional layers for managing different patch sizes
    total_upscale_factor = int(log2(cf["patch_size"]))
    upscale = total_upscale_factor - 4

    if upscale >= 2: ## Patch size 16 or greater
        z3 = deconv_block(z3, z3.shape[-1], strides=2**upscale)
        z6 = deconv_block(z6, z6.shape[-1], strides=2**upscale)
        z9 = deconv_block(z9, z9.shape[-1], strides=2**upscale)
        z12 = deconv_block(z12, z12.shape[-1], strides=2**upscale)
        # print(z3.shape, z6.shape, z9.shape, z12.shape)

    if upscale < 0: ## Patch size less than 16
        p = 2**abs(upscale)
        z3 = L.MaxPool2D((p, p))(z3)
        z6 = L.MaxPool2D((p, p))(z6)
        z9 = L.MaxPool2D((p, p))(z9)
        z12 = L.MaxPool2D((p, p))(z12)

    ## Decoder 1
    x = deconv_block(z12, 128)

    s = deconv_block(z9, 128)
    s = conv_block(s, 128)

    x = L.Concatenate()([x, s])
    x = channel_attention(x)
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    out1 = L.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    ## Decoder 2
    x = deconv_block(x, 64)

    s = deconv_block(z6, 64)
    s = conv_block(s, 64)
    s = deconv_block(s, 64)
    s = conv_block(s, 64)

    x = L.Concatenate()([x, s])
    x = channel_attention(x)
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    out2 = L.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    ## Decoder 3
    x = deconv_block(x, 32)

    s = deconv_block(z3, 32)
    s = conv_block(s, 32)
    s = deconv_block(s, 32)
    s = conv_block(s, 32)
    s = deconv_block(s, 32)
    s = conv_block(s, 32)

    x = L.Concatenate()([x, s])
    x = channel_attention(x)
    x = conv_block(x, 32)
    x = conv_block(x, 32)
    out3 = L.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x)

    ## Decoder 4
    x = deconv_block(x, 16)

    s = conv_block(z0, 16)
    s = conv_block(s, 16)

    x = L.Concatenate()([x, s])
    x = channel_attention(x)
    x = conv_block(x, 16)
    x = conv_block(x, 16)
    final_output = L.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid", name="final_output")(x)

    out1 = L.Resizing(256, 256, interpolation="bilinear", name="out1")(out1)
    out2 = L.Resizing(256, 256, interpolation="bilinear", name="out2")(out2)
    out3 = L.Resizing(256, 256, interpolation="bilinear", name="out3")(out3)

    return Model(inputs, [out1, out2, out3, final_output], name="UNETR_2D")

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime
import cv2

smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    y_true_sum = tf.reduce_sum(y_true)
    y_pred_sum = tf.reduce_sum(y_pred)
    union = y_true_sum + y_pred_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

import tensorflow.keras.backend as K
import cv2
from glob import glob
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
import os
from numpy import log2
from tensorflow.keras.models import load_model

def dice_loss(y_true, y_pred, smooth=1e-15):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

import tensorflow as tf
import cv2
import numpy as np

def preprocess(image_path, mask_path, image_size=256):
    image = cv2.imread(image_path.decode(), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image_size, image_size))
    
    if image.max() > 1.0:
        image = image / 255.0

    mask = cv2.imread(mask_path.decode(), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (image_size, image_size))

    if mask.max() > 1.0:
        mask = mask / 255.0
    
    # Check ratio of zeros and ones
    zero_ratio = np.mean(mask == 0)
    one_ratio = np.mean(mask == 1)

    # Invert if zeros are majority
    if zero_ratio > one_ratio:
        mask = 1 - mask
    mask = np.expand_dims(mask, axis=-1)

    return image.astype(np.float32), mask.astype(np.float32)

def tf_preprocess(image_path, mask_path):
    image, mask = tf.numpy_function(
        preprocess, [image_path, mask_path],
        [tf.float32, tf.float32]
    )
    image.set_shape([cf["image_size"], cf["image_size"], cf["num_channels"]])
    mask.set_shape([cf["image_size"], cf["image_size"], 1])
    return image, mask

def prepare_dataset(image_paths, mask_paths, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def load_dataset(path, split=0.1):
    train_data_path = os.path.join(path, "ISBI2016_ISIC_Part1_Training_Data")
    ground_truth_path = os.path.join(path, "ISBI2016_ISIC_Part1_Training_GroundTruth")
    test_data_path = os.path.join(path, "ISBI2016_ISIC_Part1_Test_Data")
    test_ground_truth_path = os.path.join(path, "ISBI2016_ISIC_Part1_Test_GroundTruth")

    # Get all training image and mask paths
    train_images = sorted(glob(os.path.join(train_data_path, "*.jpg")))
    train_masks = sorted(glob(os.path.join(ground_truth_path, "*_Segmentation.png")))

    # Get all test image and mask paths
    test_images = sorted(glob(os.path.join(test_data_path, "*.jpg")))
    test_masks = sorted(glob(os.path.join(test_ground_truth_path, "*_Segmentation.png")))
    """ Spliting the data into training and testing """
    split_size = int(len(train_images) * split)
    train_x, valid_x = train_test_split(train_images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_masks, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_images, test_masks)

def load_dataset_refuge(path,path1,path2 ):
    """ Loading the images and masks """
    train_x = sorted(glob(os.path.join(path, "images", "*.jpg")))
    valid_x=sorted(glob(os.path.join(path2, "images", "*.jpg")))
    test_x=sorted(glob(os.path.join(path1, "images", "*.jpg")))

    train_y= sorted(glob(os.path.join(path, "mask", "*.bmp")))
    valid_y=sorted(glob(os.path.join(path2, "mask", "*.png")))
    test_y=sorted(glob(os.path.join(path1, "mask", "*.bmp")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

import os
from glob import glob
from sklearn.model_selection import train_test_split

def load_dataset_origa(path, train_split=0.7, val_split=0.2, test_split=0.1, random_state=42):
    """Loading all images and masks for ORIGA dataset with train, validation, and test splits."""
    images = sorted(glob(os.path.join(path, "Images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "Masks", "*.png")))
    assert len(images) == len(masks), "Number of images and masks must match"
    train_x, temp_x, train_y, temp_y = train_test_split(
        images, masks, test_size=(1 - train_split), random_state=random_state
    )
    val_ratio = val_split / (val_split + test_split)
    valid_x, test_x, valid_y, test_y = train_test_split(
        temp_x, temp_y, test_size=(1 - val_ratio), random_state=random_state
    )
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


""" UNETR  Configration """
cf = {}
cf["image_size"] = 256
cf["num_channels"] = 3
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_patches"] = (cf["image_size"]**2)//(cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"]*cf["patch_size"]*cf["num_channels"]
)
cf["num_query_heads"] = 8
cf["num_key_value_heads"] = 4
cf["head_dim"] = cf["hidden_dim"] // cf["num_query_heads"]


from tensorflow.keras.optimizers import Adam
batch_size = 8
lr = 1e-3
num_epochs = 150

## ISIC
# dataset_path = "isic/isic 2016"
# (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

## REFUGE2
# dataset_path = "REFUGE2/train"
# dataset_path1 = "REFUGE2/test"
# dataset_path2 = "REFUGE2/val"
# (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset_refuge(dataset_path,dataset_path1,dataset_path2)


##ORIGA
dataset_name = 'ORIGA'
dataset_path = "datasets/ORIGA"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset_origa(dataset_path)

train_ds = prepare_dataset(train_x, train_y, batch_size=batch_size)
val_ds = prepare_dataset(valid_x, valid_y, shuffle=False)


def format_multi_outputs(image, mask):
    image = tf.reshape(image, [tf.shape(image)[0], cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"]])
    return image, {
        "out1": mask,
        "out2": mask,
        "out3": mask,
        "final_output": mask,
    }
train_ds = train_ds.map(format_multi_outputs)
val_ds = val_ds.map(format_multi_outputs)



model = build_unetr_2d(cf)

losses = {
    "out1": bce_dice_loss,
    "out2": bce_dice_loss,
    "out3": bce_dice_loss,
    "final_output": bce_dice_loss,
}

loss_weights = {
    "out1": 0.2,
    "out2": 0.3,
    "out3": 0.5,
    "final_output": 1.0,
}

metrics = {
    "out1": ["accuracy", dice_coef, iou],
    "out2": ["accuracy", dice_coef, iou],
    "out3": ["accuracy", dice_coef, iou],
    "final_output": ["accuracy", dice_coef, iou],
}



model.compile(optimizer=Adam(lr), loss=losses,
              loss_weights=loss_weights,
              metrics=metrics
              )

model.summary()

csv_logger = CSVLogger(f"training_metrics{dataset_name}.log", append=True)

checkpoint_cb = ModelCheckpoint(
    filepath='best_unetr_model.h5',
    monitor='val_final_output_dice_coef',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    verbose=2,
    callbacks=[checkpoint_cb, csv_logger]
)


test_ds = prepare_dataset(test_x, test_y, shuffle=False)
test_ds = test_ds.map(format_multi_outputs)

print("\n🔹 Evaluating Last Checkpoint:")
last_results = model.evaluate(test_ds)
print(dict(zip(model.metrics_names, last_results)))

print("\nEvaluating Best Checkpoint:")
best_model = load_model('best_unetr_model.h5', compile=False)
best_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics) 
best_results = best_model.evaluate(test_ds)
print(dict(zip(best_model.metrics_names, best_results)))



