import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import cf
from model.ModUnetr2D import build_unetr_2d
from losses import bce_dice_loss, dice_coef, iou
from data.REFUGE2_cup import load_dataset,tf_dataset
from data.REFUGE2_disc import load_dataset,tf_dataset
if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 8
    lr = 1e-3
    num_epochs = 150
    
    # REFUGE2
    dataset_path = "./REFUGE2/train"
    dataset_path1 = "./REFUGE2/test"
    dataset_path2 = "./REFUGE2/val"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path,dataset_path1,dataset_path2)
    

    def format_multi_outputs(image, mask):
        image = tf.reshape(image, [tf.shape(image)[0], cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"]])
        return image, {
            "out1": mask,
            "out2": mask,
            "out3": mask,
            "final_output": mask,
        }

    train_dataset = train_dataset.map(format_multi_outputs)
    valid_dataset = valid_dataset.map(format_multi_outputs)
    test_dataset  = test_dataset.map(format_multi_outputs)

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
        "out1": [dice_coef, iou],
        "out2": [dice_coef, iou],
        "out3": [dice_coef, iou],
        "final_output": [dice_coef, iou],
    }

    model.compile(optimizer=Adam(lr), loss=losses, loss_weights=loss_weights, metrics=metrics)
    model.summary()
    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(
    "Modunetr_model.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
    )
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            checkpoint
        ]
    )

# from evaluate import run_evaluation
# run_evaluation(model, train_dataset, valid_dataset, test_dataset)
