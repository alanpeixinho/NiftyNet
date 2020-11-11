from __future__ import absolute_import, print_function

import unittest

import tensorflow as tf
from tensorflow.keras import regularizers

from niftynet.network.resnet import ResNet
from tests.niftynet_testcase import NiftyNetTestCase

class ResNet3DTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 8, 16, 32, 1)
        x = tf.ones(input_shape)

        resnet_instance = ResNet(num_classes=160)
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 8, 16, 1)
        x = tf.ones(input_shape)

        resnet_instance = ResNet(num_classes=160)
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)

    def test_3d_reg_shape(self):
        input_shape = (2, 8, 16, 24, 1)
        x = tf.ones(input_shape)

        resnet_instance = ResNet(num_classes=160,
                               w_regularizer=regularizers.L2(0.4))
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 8, 16, 1)
        x = tf.ones(input_shape)

        resnet_instance = ResNet(num_classes=160,
                               w_regularizer=regularizers.L2(0.4))
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)


if __name__ == "__main__":
    tf.test.main()
