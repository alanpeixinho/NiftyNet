import tensorflow as tf

from niftynet.layer.bn import BNLayer
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer, ConvLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.upsample import UpSampleLayer


class FastSCNN2D(BaseNet):
    """
    Fast-SCNN: Fast Semantic Segmentation Network (2019)
    https://arxiv.org/pdf/1902.04502.pdf
    """
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='Fast-SCNN'):
        super(FastSCNN2D, self).__init__(name=name)
        self.num_classes = num_classes
        self.acti_func = acti_func
        self.w_regularizer = w_regularizer
        self.w_initializer = w_initializer

    def layer_op(self, images, is_training=True, layer_id=-1, keep_prob=0.7, **unused_kwargs):

        print('learning downsample ...')
        

        # >>>>>>>>>>>>>>>> learning down sample
        lds1 = ConvolutionalLayer(32, conv_type='REGULAR', kernel_size=3, stride=2,
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)
        lds2 = ConvolutionalLayer(48, conv_type='SEPARABLE_2D', kernel_size=3, stride=2,
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)
        lds3 = ConvolutionalLayer(64, conv_type='SEPARABLE_2D', kernel_size=3, stride=2)

        flow = lds1(images, is_training=is_training)
        flow = lds2(flow, is_training=is_training)
        flow = lds3(flow, is_training=is_training)

        lds = flow

        # >>>>>>>>>>>>>>>> global feature extraction

        print('global feature extractor ...')
        

        bottle1 = SCCNBottleneckBlock(64, 3, t=6, stride=2, n=3,
                w_initializer=self.w_initializer, w_regularizer=self.w_regularizer)
        bottle2 = SCCNBottleneckBlock(96, 3, t=6, stride=2, n=3,
                        w_initializer=self.w_initializer, w_regularizer=self.w_regularizer)
        bottle3 = SCCNBottleneckBlock(128, 3, t=6, stride=1, n=3,
                        w_initializer=self.w_initializer, w_regularizer=self.w_regularizer)
        pyramid = SCNNPyramidBlock([2, 4, 6, 8],
                        w_initializer=self.w_initializer, w_regularizer=self.w_regularizer)

        flow = bottle1(flow)
        flow = bottle2(flow)
        flow = bottle3(flow)

        flow = pyramid(flow)

        gfe = flow

        # >>>>>>>>>>>>>>>> feature fusion

        print('Feature fusion ...')
        

        conv1 = ConvolutionalLayer(128, conv_type='REGULAR', kernel_size=1, padding='same', stride=1, acti_func=None,
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)

        upsample1 = tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear')
        dwconv = ConvolutionalLayer(1, conv_type='DEPTHWISE_2D', kernel_size=3,
                                    stride=1,
                                    padding='same',
                                    acti_func=self.acti_func,
                                    w_initializer=self.w_initializer,
                                    w_regularizer=self.w_regularizer)

        conv2 = ConvLayer(128, conv_type='REGULAR',
            kernel_size=1, padding='same', stride=1,
            w_initializer=self.w_initializer,
            w_regularizer=self.w_regularizer)

        bn = BNLayer()
        acti = ActiLayer(
            func=self.acti_func,
            regularizer=self.w_regularizer,
            name='acti_')

        flow1 = conv1(lds, is_training=is_training)

        flow2 = upsample1(gfe)
        flow2 = dwconv(flow2, is_training=is_training)
        flow2 = conv2(flow2)

        flow = tf.math.add(flow1, flow2)
        flow = bn(flow, is_training=is_training)
        flow = acti(flow)

        # ff = flow

        # >>>>>>>>>>>>>>>> classifier

        

        sep_conv1 = ConvolutionalLayer(128, conv_type='SEPARABLE_2D', kernel_size=3, padding='same', stride=1,
                                 name='DSConv1_classifier', acti_func=self.acti_func,
                                 w_initializer=self.w_initializer,
                                 w_regularizer=self.w_regularizer)

        sep_conv2 = ConvolutionalLayer(128, conv_type='SEPARABLE_2D', kernel_size=3, padding='same', stride=1,
                                 name='DSConv2_classifier', acti_func=self.acti_func,
                                 w_initializer=self.w_initializer,
                                 w_regularizer=self.w_regularizer)

        flow = sep_conv1(flow, is_training=is_training)
        flow = sep_conv2(flow, is_training=is_training)


        conv = ConvolutionalLayer(self.num_classes, conv_type='REGULAR',
                    kernel_size=1, padding='same', stride=1,
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)

        dropout = ActiLayer(
            func='dropout',
            regularizer=self.w_regularizer,
            name='dropout_')
        # tf.keras.layers.Dropout(0.3)
        upsample = tf.keras.layers.UpSampling2D((8, 8), interpolation='bilinear')

        flow = conv(flow, is_training=is_training)
        flow = dropout(flow, keep_prob=keep_prob)
        flow = upsample(flow)

        flow = tf.nn.softmax(flow)

        

        return flow


class SCCNBottleneckBlock(TrainableLayer):

    def __init__(self,
                 filters=8, kernel=1,
                 t=1, stride=1, n=1, w_initializer=None, w_regularizer=None, name='scnn_bottleneck'):
        super(SCCNBottleneckBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.t = t
        self.n = n
        self.w_initializer = w_initializer
        self.w_regularizer= w_regularizer

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        x = self._res_bottleneck(images, self.filters, self.kernel, self.t, self.stride, is_training=is_training)

        for i in range(1, self.n):
            x = self._res_bottleneck(x, self.filters, self.kernel, self.t, 1, True, is_training=is_training)

        return x

    def _res_bottleneck(self, inputs, filters, kernel, t, s, r=False, is_training=True):
        tchannel = tf.keras.backend.int_shape(inputs)[-1] * t
        #conv layer only (no activation or dropout)
        conv_block = ConvLayer(kernel_size=1, n_output_chns=tchannel, stride=1, padding='same',
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)

        x = conv_block(inputs)
        x = ConvolutionalLayer(kernel, stride=s, padding='same', conv_type='DEPTHWISE_2D',
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)(x, is_training=is_training)

        conv_block_2 = ConvolutionalLayer(filters, kernel_size=1, stride=1, padding='same', acti_func=None,
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)
        x = conv_block_2(x, is_training=is_training)

        if r:
            x = tf.math.add(x, inputs)
        return x


class SCNNPyramidBlock(TrainableLayer):

    def __init__(self, bin_sizes=[2, 4, 6, 8], w_initializer = None, w_regularizer = None, name='scnn_pyramid'):
        super(SCNNPyramidBlock, self).__init__(name=name)
        self.bin_sizes = bin_sizes
        self.w_initializer = w_initializer
        self.w_regularizer = w_regularizer

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        
        concat_list = [images]

        w, h = images.shape[1], images.shape[2]

        for bin_size in self.bin_sizes:
            x = tf.layers.AveragePooling2D(pool_size=(w // bin_size, h // bin_size),
                                                 strides=(w // bin_size, h // bin_size))(images)
            x = ConvLayer(128, kernel_size=3, stride=2, padding='same',
                    w_initializer=self.w_initializer,
                    w_regularizer=self.w_regularizer)(x)
            x = tf.image.resize_images(x, (w,h))

            concat_list.append(x)

        x = tf.concat(concat_list, axis=-1)

        return x