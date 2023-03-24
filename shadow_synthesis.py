import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from networks import build_shadow_generator
import typer

app = typer.Typer()


def generate_shadows(image_folder: str, shadow_folder: str, output_folder: str, pretrain_model_path: str,
                     create_dataset: bool):
    # create output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if create_dataset:
        Path(os.path.join(output_folder, 'train_A')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_folder, 'train_B')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(output_folder, 'train_C')).mkdir(parents=True, exist_ok=True)
    tf.compat.v1.disable_eager_execution()
    # tf.compat.v1.reset_default_graph()

    channel = 64
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        mask = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1])

        shadowed_image = build_shadow_generator(tf.concat([input, mask], axis=3), channel) * input

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    idtd_ckpt = tf.train.get_checkpoint_state(pretrain_model_path)
    saver_restore = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
    print('loaded ' + idtd_ckpt.model_checkpoint_path)
    saver_restore.restore(sess, idtd_ckpt.model_checkpoint_path)

    image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if
                   x.endswith('.jpg') or x.endswith('.png')]
    mask_paths = [os.path.join(shadow_folder, x) for x in os.listdir(shadow_folder) if
                  x.endswith('.png') or x.endswith('.jpg')]

    for img_path in image_paths:
        mask_path = random.choice(mask_paths)

        iminput, immask = cv2.resize(cv2.imread(img_path, 1), (640, 480)), cv2.resize(cv2.imread(mask_path, 1),
                                                                                      (640, 480))
        imoutput = sess.run(shadowed_image, feed_dict={input: np.expand_dims(iminput / 255., axis=0),
                                                       mask: np.expand_dims(immask[:, :, 0:1] / 255., axis=0)})

        imshadow = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput, 0.0), 1.0)) * 255.0)

        output_filename = os.path.join(output_folder, os.path.basename(img_path))
        # copy mask to train_B with name of image
        if create_dataset:
            cv2.imwrite(os.path.join(output_folder, 'train_A', os.path.basename(img_path)), imshadow[..., ::-1])
            cv2.imwrite(os.path.join(output_folder, 'train_B', os.path.basename(img_path)), immask[..., ::-1])
            cv2.imwrite(os.path.join(output_folder, 'train_C', os.path.basename(img_path)), iminput[..., ::-1])
        else:
            cv2.imwrite(output_filename, imshadow[..., ::-1])


@app.command()
def main(
        image_folder: str = typer.Argument(..., help="Path to the sample images"),
        shadow_folder: str = typer.Argument(..., help="Path to the sample shadows"),
        output_folder: str = typer.Argument(..., help="Path to the output images"),
        pretrain_model_path: str = typer.Argument("ss", help="Path to the pretrained model"),
        create_dataset: bool = typer.Option(False, help="Create dataset"),
):
    generate_shadows(image_folder, shadow_folder, output_folder, pretrain_model_path, create_dataset)


if __name__ == "__main__":
    app()
