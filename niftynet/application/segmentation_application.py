# -*- coding: utf-8 -*-
import tensorflow as tf

import numpy
from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.sampler_balanced_v2 import BalancedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.utilities.util_common import color_labels, generate_color_palette
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.layer.rgb_histogram_equilisation import \
    RGBHistogramEquilisationLayer
from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer

SUPPORTED_INPUT = set(
    ['image', 'label', 'weight', 'sampler', 'inferred', 'value'])

class SegmentationApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, action):
        super(SegmentationApplication, self).__init__()
        tf.compat.v1.logging.info('starting segmentation application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.__im_summary_type_dict = {'xy': 'image3_axial', 'xz': 'image3_coronal', 'yz': 'image3_sagittal'}

        self.data_param = None
        self.segmentation_param = None
        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator),
            'balanced': (self.initialise_balanced_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):

        print('initialise dataset loader')

        self.data_param = data_param
        self.segmentation_param = task_param

        # initialise input image readers
        if self.is_training:
            reader_names = ('image', 'label', 'weight', 'sampler')
        elif self.is_inference:
            # in the inference process use `image` input only
            reader_names = ('image',)
        elif self.is_evaluation:
            reader_names = ('image', 'label', 'inferred')
        elif self.is_export:
            reader_names = ('image',)
        else:
            tf.compat.v1.logging.fatal(
                'Action `%s` not supported. Expected one of %s',
                self.action, self.SUPPORTED_PHASES)
            raise ValueError
        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        self.readers = [
            ImageReader(reader_names).initialise(
                data_param, task_param, file_list) for file_list in file_lists]

        # initialise input preprocessing layers
        foreground_masking_layer = BinaryMaskingLayer(
            type_str=self.net_param.foreground_type,
            multimod_fusion=self.net_param.multimod_foreground_type,
            threshold=0.0) \
            if self.net_param.normalise_foreground_only else None
        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image', binary_masking_func=foreground_masking_layer) \
            if self.net_param.whitening else None
        histogram_normaliser = HistogramNormalisationLayer(
            image_name='image',
            modalities=vars(task_param).get('image'),
            model_filename=self.net_param.histogram_ref_file,
            binary_masking_func=foreground_masking_layer,
            norm_type=self.net_param.norm_type,
            cutoff=self.net_param.cutoff,
            name='hist_norm_layer') \
            if (self.net_param.histogram_ref_file and
                self.net_param.normalisation) else None
        rgb_normaliser = RGBHistogramEquilisationLayer(
            image_name='image',
            name='rbg_norm_layer') if self.net_param.rgb_normalisation else None
        label_normalisers = None
        if self.net_param.histogram_ref_file and \
                task_param.label_normalisation:
            label_normalisers = [DiscreteLabelNormalisationLayer(
                image_name='label',
                modalities=vars(task_param).get('label'),
                model_filename=self.net_param.histogram_ref_file)]
            if self.is_evaluation:
                label_normalisers.append(
                    DiscreteLabelNormalisationLayer(
                        image_name='inferred',
                        modalities=vars(task_param).get('inferred'),
                        model_filename=self.net_param.histogram_ref_file))
                label_normalisers[-1].key = label_normalisers[0].key

        normalisation_layers = []
        if histogram_normaliser is not None:
            normalisation_layers.append(histogram_normaliser)
        if rgb_normaliser is not None:
            normalisation_layers.append(rgb_normaliser)
        if mean_var_normaliser is not None:
            normalisation_layers.append(mean_var_normaliser)
        if task_param.label_normalisation and \
                (self.is_training or not task_param.output_prob):
            normalisation_layers.extend(label_normalisers)

        volume_padding_layer = [PadLayer(
            image_name=SUPPORTED_INPUT,
            border=self.net_param.volume_padding_size,
            mode=self.net_param.volume_padding_mode,
            pad_to=self.net_param.volume_padding_to_size)
        ]
        # initialise training data augmentation layers
        augmentation_layers = []
        if self.is_training:
            train_param = self.action_param
            self.patience = train_param.patience
            self.mode = self.action_param.early_stopping_mode
            if train_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=train_param.random_flipping_axes))
            if train_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=train_param.scaling_percentage[0],
                    max_percentage=train_param.scaling_percentage[1],
                    antialiasing=train_param.antialiasing,
                    isotropic=train_param.isotropic_scaling))
            if train_param.rotation_angle or \
                    train_param.rotation_angle_x or \
                    train_param.rotation_angle_y or \
                    train_param.rotation_angle_z:
                rotation_layer = RandomRotationLayer()
                if train_param.rotation_angle:
                    rotation_layer.init_uniform_angle(
                        train_param.rotation_angle)
                else:
                    rotation_layer.init_non_uniform_angle(
                        train_param.rotation_angle_x,
                        train_param.rotation_angle_y,
                        train_param.rotation_angle_z)
                augmentation_layers.append(rotation_layer)
            if train_param.do_elastic_deformation:
                spatial_rank = list(self.readers[0].spatial_ranks.values())[0]
                augmentation_layers.append(RandomElasticDeformationLayer(
                    spatial_rank=spatial_rank,
                    num_controlpoints=train_param.num_ctrl_points,
                    std_deformation_sigma=train_param.deformation_sigma,
                    proportion_to_augment=train_param.proportion_to_deform))

        # only add augmentation to first reader (not validation reader)
        self.readers[0].add_preprocessing_layers(
            volume_padding_layer + normalisation_layers + augmentation_layers)

        for reader in self.readers[1:]:
            reader.add_preprocessing_layers(
                volume_padding_layer + normalisation_layers)

        # Checking num_classes is set correctly
        if self.segmentation_param.num_classes <= 1:
            raise ValueError("Number of classes must be at least 2 for segmentation")
        for preprocessor in self.readers[0].preprocessors:
            if preprocessor.name == 'label_norm':
                if len(preprocessor.label_map[preprocessor.key[0]]) != self.segmentation_param.num_classes:
                    raise ValueError("Number of unique labels must be equal to "
                                     "number of classes (check histogram_ref file)")

    def initialise_uniform_sampler(self):
        self.sampler = [[UniformSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_weighted_sampler(self):
        self.sampler = [[WeightedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle=self.is_training,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_sampler(self):
        self.sampler = [[GridSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_balanced_sampler(self):
        self.sampler = [[BalancedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix,
            fill_constant=self.action_param.fill_constant)

    def initialise_resize_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        elif self.is_inference:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()
        elif self.is_export:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()


    def initialise_aggregator(self):
        self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]()

    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.keras import regularizers
            w_regularizer = regularizers.L2(decay)
            b_regularizer = regularizers.L2(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.keras import regularizers
            w_regularizer = regularizers.L1(decay)
            b_regularizer = regularizers.L1(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.segmentation_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        print('connect data and network')
        def switch_sampler(for_training):
            with tf.compat.v1.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        def mixup_switch_sampler(for_training):
            # get first set of samples
            d_dict = switch_sampler(for_training=for_training)

            mix_fields = ('image', 'weight', 'label')

            if not for_training:
                with tf.compat.v1.name_scope('nomix'):
                    # ensure label is appropriate for dense loss functions
                    ground_truth = tf.cast(d_dict['label'], tf.int32)
                    one_hot = tf.one_hot(tf.squeeze(ground_truth, axis=-1),
                                         depth=self.segmentation_param.num_classes)
                    d_dict['label'] = one_hot
            else:
                with tf.compat.v1.name_scope('mixup'):
                    # get the mixing parameter from the Beta distribution
                    alpha = self.segmentation_param.mixup_alpha
                    beta = tf.compat.v1.distributions.Beta(alpha, alpha)  # 1, 1: uniform:
                    rand_frac = beta.sample()

                    # get another minibatch
                    d_dict_to_mix = switch_sampler(for_training=True)

                    # look at binarised labels: sort them
                    if self.segmentation_param.mix_match:
                        # sum up the positive labels to sort by their volumes
                        inds1 = tf.argsort(tf.map_fn(tf.reduce_sum, tf.cast(d_dict['label'], tf.int64)))
                        inds2 = tf.argsort(tf.map_fn(tf.reduce_sum, tf.cast(d_dict_to_mix['label'] > 0, tf.int64)))
                        for field in [field for field in mix_fields if field in d_dict]:
                            d_dict[field] = tf.gather(d_dict[field], indices=inds1)
                            # note: sorted for opposite directions for d_dict_to_mix
                            d_dict_to_mix[field] = tf.gather(d_dict_to_mix[field], indices=inds2[::-1])

                    # making the labels dense and one-hot
                    for d in (d_dict, d_dict_to_mix):
                        ground_truth = tf.cast(d['label'], tf.int32)
                        one_hot = tf.one_hot(tf.squeeze(ground_truth, axis=-1),
                                             depth=self.segmentation_param.num_classes)
                        d['label'] = one_hot

                    # do the mixing for any fields that are relevant and present
                    mixed_up = {field: d_dict[field] * rand_frac + d_dict_to_mix[field] * (1 - rand_frac) for field
                                in mix_fields if field in d_dict}
                    # reassign all relevant values in d_dict
                    d_dict.update(mixed_up)

            return d_dict

        if self.is_training:
            if not self.segmentation_param.do_mixup:
                data_dict = tf.cond(pred=tf.logical_not(self.is_validation),
                                    true_fn=lambda: switch_sampler(for_training=True),
                                    false_fn=lambda: switch_sampler(for_training=False))
            else:
                # mix up the samples if not in validation phase
                data_dict = tf.cond(pred=tf.logical_not(self.is_validation),
                                    true_fn=lambda: mixup_switch_sampler(for_training=True),
                                    false_fn=lambda: mixup_switch_sampler(for_training=False))  # don't mix the validation

            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            net_out = self.net(image, **net_args)

            with tf.compat.v1.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type,
                softmax=self.segmentation_param.softmax)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))
            reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    input_tensor=[tf.reduce_mean(input_tensor=reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
            else:
                loss = data_loss

            # Get all vars
            to_optimise = tf.compat.v1.trainable_variables()
            vars_to_freeze = \
                self.action_param.vars_to_freeze or \
                self.action_param.vars_to_restore
            if vars_to_freeze:
                import re
                var_regex = re.compile(vars_to_freeze)
                # Only optimise vars that are not frozen
                to_optimise = \
                    [v for v in to_optimise if not var_regex.search(v.name)]
                tf.compat.v1.logging.info(
                    "Optimizing %d out of %d trainable variables, "
                    "the other variables fixed (--vars_to_freeze %s)",
                    len(to_optimise),
                    len(tf.compat.v1.trainable_variables()),
                    vars_to_freeze)

            grads = self.optimiser.compute_gradients(
                loss, var_list=to_optimise)

            self.total_loss = loss

            # collecting gradients variables
            gradients_collector.add_to_collection([grads])

            # collecting output variables
            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
#             outputs_collector.add_to_collection(
#                 var=data_loss, name='loss',
#                 average_over_devices=False, collection=CONSOLE)
#             outputs_collector.add_to_collection(
#                 var=data_loss, name='loss',
#                 average_over_devices=True, summary_type='scalar',
#                 collection=TF_SUMMARIES)

            #import pdb; pdb.set_trace()

            if image.shape[3]>1:#is 3d image
                axes = ['xy', 'yz', 'xz']
            else:
                axes = ['xy']
            for axis in axes:
                self.tensorboard_preview_collectors(outputs_collector, image, data_dict.get('label', None), net_out, axis)

#             net_out = tf.Print(net_out, [net_out], message='Net output: ')



            # outputs_collector.add_to_collection(
            #    var=image, name='image',
            #    average_over_devices=False,
            #    collection=NETWORK_OUTPUT)

            # outputs_collector.add_to_collection(
            #    var=tf.reduce_mean(image), name='mean_image',
            #    average_over_devices=False, summary_type='scalar',
            #    collection=CONSOLE)
        elif self.is_inference:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            net_out = self.net(image, **net_args)

            output_prob = self.segmentation_param.output_prob
            num_classes = self.segmentation_param.num_classes
            if output_prob and num_classes > 1:
                post_process_layer = PostProcessingLayer(
                    'SOFTMAX', num_classes=num_classes)
            elif not output_prob and num_classes > 1:
                post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=num_classes)
            else:
                post_process_layer = PostProcessingLayer(
                    'IDENTITY', num_classes=num_classes)
            net_out = post_process_layer(net_out)

            outputs_collector.add_to_collection(
                var=net_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()
        elif self.is_export:
            data_dict = switch_sampler(for_training=False)
            output_prob = self.segmentation_param.output_prob
            num_classes = self.segmentation_param.num_classes
            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            
            net_out = self.net(image, **net_args)
            post_process_layer = PostProcessingLayer(
                'SOFTMAX', num_classes=num_classes)
            net_out = post_process_layer(net_out)
            self.initialise_aggregator()


    def interpret_output(self, batch_output):
        if self.is_inference:
            return self.output_decoder.decode_batch(
                {'window_seg': batch_output['window']},
                batch_output['location'])

        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        self.evaluator = SegmentationEvaluator(self.readers[0],
                                               self.segmentation_param,
                                               eval_param)

    def add_inferred_output(self, data_param, task_param):
        return self.add_inferred_output_like(data_param, task_param, 'label')

    def tensorboard_image_normalize(self, x):
        immin, immax = tf.reduce_min(input_tensor=x), tf.reduce_max(input_tensor=x)
        log = ((x - immin) / immax) * 255
        return log

    def tensorboard_add_class(self, outputs_collector, ground_truth, output, image, c, slice_idx, middle_slice_idx, axis = 'xy'):
        log_out = self.tensorboard_image_normalize(output[..., c:c+1])
        log_gt = tf.cast(tf.math.equal(ground_truth , c), dtype='float32') * 255
        log_image = self.tensorboard_image_normalize(image)

        is_3d = len(output.shape) >= 5  # is 3d

        if is_3d:
            print('add slice {} ...'.format(middle_slice_idx))

            outputs_collector.add_to_collection(
                var=log_out[slice_idx],
                name='class_{}/{}/slice/segmentation'.format(c, axis),
                average_over_devices=False, summary_type='image',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=log_gt[slice_idx],
                name='class_{}/{}/slice/ground_truth'.format(c, axis),
                average_over_devices=False, summary_type='image',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=log_image[slice_idx], summary_type='image',
                average_over_devices=False, name='class_{}/{}/slice/input'.format(c, axis),
                collection=TF_SUMMARIES)
        else:
            outputs_collector.add_to_collection(
                var=log_out[:1,...], name='class_{}/{}/segmentation'.format(c, axis),
                average_over_devices=False, summary_type='image',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=log_gt[:1,...], name='class_{}/{}/ground_truth'.format(c, axis),
                average_over_devices=False, summary_type='image',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=log_image[:1, ...], summary_type='image',
                average_over_devices=False, name='class_{}/{}/input'.format(c, axis),
                collection=TF_SUMMARIES)


    def tensorboard_preview_collectors(self, outputs_collector, image, ground_truth, output, axis = 'xy'):

        nclasses = self.segmentation_param.num_classes
        print("Number of classes: ", nclasses)

        is_3d = len(output.shape) >= 5  # is 3d

        middle_slice_idx = slice_idx = None
        if is_3d:
            if axis == 'xy':
                middle_slice_idx = image.shape[3] // 2
                slice_idx = numpy.index_exp[:1, :, :, middle_slice_idx, ...]#show only middle slice of first image on batch
            elif axis == 'yz':
                middle_slice_idx = image.shape[1] // 2
                slice_idx = numpy.index_exp[:1, middle_slice_idx, ...]
            else:#xz
                middle_slice_idx = image.shape[2] // 2
                slice_idx = numpy.index_exp[:1, :, middle_slice_idx, ...]


        for c in range(nclasses):
            print("Log variables for class {}".format(c))
            self.tensorboard_add_class(outputs_collector, ground_truth, output, image, c, slice_idx, middle_slice_idx, axis)
