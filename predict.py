import os

import cv2
import numpy as np
import tensorflow as tf
import typer

from networks import build_aggasatt_joint

tf.compat.v1.disable_eager_execution()


def predict_shadow_removal(input_folder: str, output_folder: str):
    vgg19_path = 'Models/imagenet-vgg-verydeep-19.mat'
    pretrain_model_path = 'Models/srdplus-pretrained/'

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        shadow_free_image = build_aggasatt_joint(input, 64, vgg19_path)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    idtd_ckpt = tf.train.get_checkpoint_state(pretrain_model_path)
    saver_restore = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
    print('loaded ' + idtd_ckpt.model_checkpoint_path)
    saver_restore.restore(sess, idtd_ckpt.model_checkpoint_path)

    os.makedirs(output_folder, exist_ok=True)

    for img_path in [os.path.join(input_folder, x) for x in os.listdir(input_folder) if
                     x.endswith('.jpg') or x.endswith('.png')]:
        img_name = os.path.basename(img_path)
        iminput = cv2.imread(img_path, -1)
        iminput = cv2.resize(iminput, (512, 512))
        imoutput = sess.run(shadow_free_image, feed_dict={input: np.expand_dims(iminput / 255., axis=0)})
        immask = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput[1], 0.0), 1.0)) * 255.0)

        imremoval = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput[0], 0.0), 1.0)) * 255.0)
        cv2.imwrite(os.path.join(output_folder, img_name), imremoval)


def main(input_folder: str = typer.Argument(..., help='Input folder'),
         output_folder: str = typer.Argument(..., help='Output folder'),
         # mask_folder: str = typer.Argument(None, help='Mask folder')
         ):
    # if not mask_folder:
    #     mask_folder = None

    predict_shadow_removal(input_folder, output_folder)


if __name__ == '__main__':
    typer.run(main)
