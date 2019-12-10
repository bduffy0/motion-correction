# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes

from niftynet.layer.base_layer import Layer
from niftynet.utilities.util_common import look_up_operations
from niftynet.utilities.util_common import otsu_threshold

"""

gets the mask from the input_data binarizes it by thresholding at zero and returns it as the second output arg

Note: this doesn't apply the mask to the image - just returns it for NiftyNet preprocesing

"""


class MaskLayer(Layer):
    def __init__(self,image_name,mask_name):
        super(MaskLayer, self).__init__(name='binary_masking')
        self.image_name = image_name
        self.mask_name = mask_name

    def layer_op(self, input_data,mask=None):

        if self.mask_name in input_data:
            mask = dict()
            mask[self.image_name] = input_data[self.mask_name]>0
        else:
            print('warning no mask provided')

        return input_data, mask