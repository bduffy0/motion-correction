from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_identity import WindowAsImageAggregator
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.layer.pad import PadLayer

from layer.crop import CropLayer
from layer.motion_sim_layer import MotionSimLayer
from layer.mask import MaskLayer

from niftynet.layer.loss_regression import rmse_loss
from utils.util import create_image_summary
from niftynet.io.image_reader import ImageReader

SUPPORTED_INPUT = set(['image','mask'])

class Regress(BaseApplication):

    """
    Regression Application for NiftyNet motion simulation

    Requires: custom user_parameters_custom.py file for allowable config parameters

    - Motion simulation is carried out during preprocessing using MotionSimLayer
    - The loss function is weighted using the provided mask
    - Image is cropped during preprocessing based on the CropLayer
    - If input is cropped smaller than the spatial_window_size - will zero-pad to the window size
    - The 5th dimension of 'image' contains the ground-truth and corrupted image respectively

    """
    REQUIRED_CONFIG_SECTION = "REGRESSION_MOT_SIM"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        print('action = {}'.format(action))
        self.net_param = net_param
        self.action_param = action_param
        self.action = action
        self.data_param = None
        self.custom_param = None
        self.current_lr = action_param.lr
        self.SUPPORTED_SAMPLING = {
        'uniform': (self.initialise_uniform_sampler,
                    self.initialise_grid_sampler,
                    self.initialise_grid_aggregator),
        'resize': (self.initialise_resize_sampler,
                   self.initialise_resize_sampler,
                   self.initialise_resize_aggregator),
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.custom_param = task_param

        file_lists = self.get_file_lists(data_partitioner)
        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.readers = []
            for file_list in file_lists:
                reader = ImageReader(['image','mask'])
                reader.initialise(data_param, task_param, file_list)
                self.readers.append(reader)
        else:
            inference_reader = ImageReader(['image','mask']) # for masking at inference
            # inference_reader = ImageReader(['image'])

            inference_reader.initialise(data_param, task_param, file_lists[0])
            self.readers = [inference_reader]

        foreground_masking_layer = None
        if self.net_param.normalise_foreground_only:
            foreground_masking_layer = BinaryMaskingLayer(
                type_str=self.net_param.foreground_type,
                multimod_fusion=self.net_param.multimod_foreground_type,
                threshold=0.0)

        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image',
            binary_masking_func=foreground_masking_layer)
        histogram_normaliser = None
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)

        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=self.action_param.random_flipping_axes))
            if self.action_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=self.action_param.scaling_percentage[0],
                    max_percentage=self.action_param.scaling_percentage[1]))
            if self.action_param.rotation_angle:
                augmentation_layers.append(RandomRotationLayer())
                augmentation_layers[-1].init_uniform_angle(
                    self.action_param.rotation_angle)

            motionsimlayer = MotionSimLayer('image',name='motion_sim',
                                            std_rotation_angle=self.custom_param.mot_rotation_angle,
                                            std_translation=self.custom_param.mot_translation,
                                            corrupt_pct=self.custom_param.corrupt_pct,
                                            freq_encoding_dim=self.custom_param.freq_encoding_dim,
                                            preserve_center_pct=self.custom_param.preserve_pct,
                                            proc_scale=self.custom_param.proc_scale,
                                            nufft=self.custom_param.nufft,
                                            apply_mask=False)

            for reader in self.readers:
                reader.add_preprocessing_layers(MaskLayer('image','mask'))

            for reader in self.readers:
                reader.add_preprocessing_layers(motionsimlayer)

            crop_layer = CropLayer('image', name='crop_layer',border_size=8,
                                   min_image_size=[int(x*1.1) for x in self.data_param['image'].spatial_window_size])

            for reader in self.readers:
                reader.add_preprocessing_layers(crop_layer)

            for reader in self.readers:
                reader.add_preprocessing_layers(
                    normalisation_layers + augmentation_layers)

        else:

            for reader in self.readers:
                reader.add_preprocessing_layers(
                    MaskLayer('image','mask'))

            for reader in self.readers:
                reader.add_preprocessing_layers(
                    normalisation_layers)

        volume_padding_layer = []
        volume_padding_layer.append(PadLayer(
                                    image_name=SUPPORTED_INPUT,border=(0,),
                                    pad_to=[x+self.net_param.volume_padding_size[i] for i,x in enumerate(self.data_param['image'].spatial_window_size)]))

        for reader in self.readers:
            reader.add_preprocessing_layers(volume_padding_layer)

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            shuffle=self.is_training,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_uniform_sampler(self):
        self.sampler = [[UniformSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_sampler(self):

        self.sampler = [[GridSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

    def initialise_resize_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        else:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()

    def initialise_aggregator(self):
        self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]()

    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=1,
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
        if self.is_training:
            def switch_sampler(for_training):
                with tf.name_scope('train' if for_training else 'validation'):
                    sampler = self.get_sampler()[0][0 if for_training else -1]
                    return sampler.pop_batch_op()

            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(for_training=True)

            images = data_dict['image']
            mask = tf.greater(data_dict['mask'],tf.constant([0.1],dtype=tf.float32))
            loss_weights = tf.cast(mask,tf.float32)

            net_output = self.net(tf.expand_dims(images[:,:,:,:,1],-1),
                                   self.is_training) # corrupted image is the 1st index

            loss = rmse_loss(net_output[0],tf.expand_dims(images[:,:,:,:,0],-1),
                              weight_map=loss_weights)

            # for tensorboard
            create_image_summary(tf.expand_dims(images[:,:,:,:,0],-1), 'gt_im')
            create_image_summary(tf.expand_dims(images[:,:,:,:,1],-1), 'corrupted_im')
            create_image_summary(net_output[0], 'corrected_im')
            create_image_summary(tf.expand_dims(images[:,:,:,:,1],-1)-net_output[0], 'difference_estimated')
            create_image_summary(tf.expand_dims(images[:,:,:,:,1],-1) - tf.expand_dims(images[:,:,:,:,0],-1), 'gt_difference')
            create_image_summary(loss_weights, 'loss_weight')

            if self.net_param.decay > 0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(l_reg) for l_reg in reg_losses])

                    loss = loss + reg_loss

            # variables to display in STDOUT

            outputs_collector.add_to_collection(
                var=loss, name='loss', average_over_devices=True,
                collection=CONSOLE)

            # variables to display in tensorboard

            outputs_collector.add_to_collection(
                var=loss, name='loss', average_over_devices=True,
                collection=TF_SUMMARIES)

            with tf.name_scope('Optimiser'):

                optimiser = OptimiserFactory.create(
                    name=self.action_param.optimiser)

                self.optimiser = optimiser.get_instance(
                    learning_rate=self.action_param.lr)

            with tf.name_scope('ComputeGradients'):

                grads = self.optimiser.compute_gradients(loss, colocate_gradients_with_ops=True)

                # add the grads back to application_driver's training_grads
                gradients_collector.add_to_collection([grads])

        else:

            data_dict = self.get_sampler()[0][0].pop_batch_op()

            net_output = self.net(data_dict['image'],self.is_training)[0]

            outputs_collector.add_to_collection(
                var=net_output,
                name='image',
                average_over_devices=False,
                collection=NETWORK_OUTPUT)

            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            self.output_decoder = WindowAsImageAggregator(
                image_reader=self.readers[0],
                output_path=self.action_param.save_seg_dir)

            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        if self.is_training:
            return True
        return self.output_decoder.decode_batch(
            {'window_image': batch_output['image']},
            batch_output['location'])

    def get_file_lists(self, data_partitioner):
        """This function pull the correct file_lists from the data partitioner
        depending on the phase
        :param data_partitioner:
                           specifies train/valid/infer splitting if needed
        :return:           list of file lists of length 2 if validation is
                           needed otherwise 1"""

        if self.is_training:
            if self.action_param.validation_every_n > 0 and \
                    data_partitioner.has_validation:
                return [data_partitioner.train_files,
                        data_partitioner.validation_files]
            else:
                return [data_partitioner.train_files]

        return [data_partitioner.inference_files]

    def set_network_gradient_op(self, gradients):
        """
        create gradient op by optimiser.apply_gradients
        this function sets ``self.gradient_op``.

        Override this function for more complex optimisations such as
        using different optimisers for sub-networks.

        :param gradients: processed gradients from the gradient_collector
        """

        self.gradient_op = [self.optimiser.apply_gradients(gradients[0])]
