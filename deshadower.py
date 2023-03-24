from __future__ import division

import os
import sys
import time

import tensorflow as tf

from networks import *
from utils import *

tf.compat.v1.disable_eager_execution()

EPS = 1e-12


class Deshadower(object):
    def __init__(self, model_path, vgg_19_path, use_gpu, hyper):
        self.vgg_19_path = vgg_19_path
        self.model = model_path
        self.hyper = hyper
        self.channel = 64
        if use_gpu < 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)
        self.setup_model()

    def setup_model(self):
        # set up the model and define the graph
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
            target = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
            gtmask = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1])

            # build the model
            self.shadow_free_image, self.predicted_mask = build_aggasatt_joint(self.input, self.channel,
                                                                               vgg_19_path=self.vgg_19_path)

            loss_mask = tf.reduce_mean(
                input_tensor=tf.keras.losses.binary_crossentropy(gtmask, tf.nn.sigmoid(self.predicted_mask)))

            # Perceptual Loss
            loss_percep = compute_percep_loss(self.shadow_free_image, target, vgg_19_path=self.vgg_19_path)
            # Adversarial Loss
            with tf.compat.v1.variable_scope("discriminator"):
                predict_real, pred_real_dict = build_discriminator(self.input, target)
            with tf.compat.v1.variable_scope("discriminator", reuse=True):
                predict_fake, pred_fake_dict = build_discriminator(self.input, self.shadow_free_image)

            d_loss = (tf.reduce_mean(
                input_tensor=-(tf.math.log(predict_real + EPS) + tf.math.log(1 - predict_fake + EPS)))) * 0.5
            g_loss = tf.reduce_mean(input_tensor=-tf.math.log(predict_fake + EPS))

            loss = loss_percep * 0.2 + loss_mask

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.model)

        print("[i] contain checkpoint: ", ckpt)
        saver_restore = tf.compat.v1.train.Saver(
            [var for var in tf.compat.v1.trainable_variables() if 'discriminator' not in var.name])
        print('loaded ' + ckpt.model_checkpoint_path)
        saver_restore.restore(self.sess, ckpt.model_checkpoint_path)

        sys.stdout.flush()

    def run(self, img):
        iminput = expand(img)
        st = time.time()
        imoutput, mask = self.sess.run([self.shadow_free_image, self.predicted_mask], feed_dict={self.input: iminput})
        print("Test time  = %.3f " % (time.time() - st))
        imoutput = decode_image(imoutput)
        mask = decode_image(mask)
        return imoutput, mask
