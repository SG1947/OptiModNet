import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

results_dir = "results"
saved_images = os.listdir(results_dir)
saved_images = [img for img in saved_images if img.endswith(('.png', '.jpg', '.jpeg'))]

# Dice and IoU functions
def dice_coef_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Set up the plotting grid
num_images = len(saved_images)
cols = 3
rows = (num_images + cols - 1) // cols

plt.figure(figsize=(15, 5 * rows))

for i, image_name in enumerate(saved_images):
    image_path = os.path.join(results_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    section = (w - 20) // 3  # 2 lines of 10 px between 3 images

    original = image[:, :section]
    mask = image[:, section+10:2*section+10]
    pred = image[:, 2*section+20:]

    # Convert to grayscale and normalize
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) / 255.0
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY) / 255.0

    # Binarize predictions
    mask_bin = (mask_gray > 0.5).astype(np.float32)
    pred_bin = (pred_gray > 0.5).astype(np.float32)

    dice = dice_coef_np(mask_bin, pred_bin)
    iou = iou_np(mask_bin, pred_bin)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    plt.title(f"{image_name}\nDice: {dice:.3f}, IoU: {iou:.3f}", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()
dice_scores = []
iou_scores = []

# Inside the loop
dice_scores.append(dice)
iou_scores.append(iou)

# After loop
print(f"\nAverage Dice Coefficient: {np.mean(dice_scores):.4f}")
print(f"Average IoU: {np.mean(iou_scores):.4f}")
