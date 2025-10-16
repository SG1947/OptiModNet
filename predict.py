import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from patchify import patchify
from model.ModUnetr2D import build_unetr_2d
from config import cf
from data.ISIC_dataset import load_dataset

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("results")

    """ Dataset """
    dataset_path = "...."  # or your dataset path
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Load model """
    model = build_unetr_2d(cf)
    model.load_weights("Modunetr_model.h5")

    """ Prediction """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
        x = image / 255.0  # Normalize the image

        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        patches = patchify(x, patch_shape, cf["patch_size"])
        patches = np.reshape(patches, cf["flat_patches_shape"])
        patches = patches.astype(np.float32)
        patches = np.expand_dims(patches, axis=0)

        #REFUGE
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]), interpolation=cv2.INTER_NEAREST)
        mask = ((mask == 0) | (mask == 128)).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)  # Create 3-channel for visualization
        pred = model.predict(patches, verbose=0)[0]
        pred = np.concatenate([pred, pred, pred], axis=-1)  # For visualization
        

        """ Save visual comparison """
        line = np.ones((cf["image_size"], 10, 3)) * 255
        cat_images = np.concatenate([image, line, mask * 255, line, pred * 255], axis=1)
        save_image_path = os.path.join("results", name)
        cv2.imwrite(save_image_path, cat_images)

        """ Save predicted mask separately """
        pred_mask_dir = "predicted_masks_gqa"
        create_dir(pred_mask_dir)

        pred_mask = (pred * 255).astype(np.uint8)
        if pred_mask.ndim == 3 and pred_mask.shape[2] == 1:
            pred_mask = pred_mask[:, :, 0]  # Optional: squeeze channel if needed

        pred_mask_path = os.path.join(pred_mask_dir, name)
        cv2.imwrite(pred_mask_path, pred_mask)


