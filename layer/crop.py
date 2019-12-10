
from __future__ import absolute_import, print_function

import numpy as np

from niftynet.layer.base_layer import Layer
import math
import time

class CropLayer(Layer):
    """
     Add crop sim layer
    """
    def __init__(self, image_name,name = 'crop',border_size=20,min_image_size=(1,1,1)):

        super(CropLayer, self).__init__(name=name)
        self.crop_coord = None
        self.imagename = image_name
        self.border_size = [border_size for i in range(3)]
        self.min_im_size = min_image_size
        self.imshape = None

    def crop_zeros(self,img_array):
        return [(np.min(tup), np.max(tup)+1) for tup in np.nonzero(img_array > 0)]

    def get_crop_coordinates(self,img_data):
        coord = self.crop_zeros(img_data)
        coord = np.array(coord)
        coord[coord < 0] = 0
        shape = [x[1]-x[0] for x in coord]
        return coord,shape

    def get_min_crop_coordinates(self):

        im_center = [math.ceil(x/2) for x in self.imshape]
        to_add = [[math.ceil(x/2),math.floor(x/2)] for x in self.min_im_size]
        coord = np.array([[im_center[i]-to_add[i][0],im_center[i]+to_add[i][1]] for i in range(len(im_center))])
        coord[coord < 0]=0

        for i in range(len(coord)):
            if coord[i][1] > self.imshape[i]:
                coord[i][1] = self.imshape[i]

        return coord

    def add_border(self,coords,border_size):
        new_coords = np.array([[coords[i][0]-border_size[i],coords[i][1]+border_size[i]] for i in range(len(coords))])

        new_coords[new_coords < 0] = 0

        for i in range(len(new_coords)):
            if new_coords[i][1] > self.imshape[i]:
                new_coords[i][1] = self.imshape[i]
        return new_coords

    def crop_image(self,img_data):
        x_, _x = self.crop_coord[0]
        y_, _y = self.crop_coord[1]
        z_, _z = self.crop_coord[2]
        img_data_out = img_data[x_:_x, y_:_y, z_:_z]
        return img_data_out

    def get_crop_dims(self):
        dims=[]
        for i in self.crop_coord:
            dims.append(i[1]-i[0])
        return dims

    def layer_op(self, input_image, mask=None):
        #todo: use input arg mask
        mask = input_image['mask'][:,:,:,0,0]

        self.imshape = mask.shape
        coords,crop_shape = self.get_crop_coordinates(mask)

        self.crop_coord = self.add_border(coords,self.border_size)

        min_crop_coords = self.get_min_crop_coordinates()
        for i in range(len(crop_shape)):
            if (crop_shape[i] + self.border_size[i])  < self.min_im_size[i]:
                self.crop_coord[i] = min_crop_coords[i]
                self.crop_coord[i] = self.add_border([self.crop_coord[i]], self.border_size)

        new_im_list = []
        for i in range(input_image[self.imagename].shape[4]):
            im = np.squeeze(input_image[self.imagename][:, :, :, 0, i])
            new_im = np.expand_dims(np.expand_dims(self.crop_image(im),-1),-1)
            new_im_list.append(new_im)

        mask_im = np.squeeze(input_image['mask'][:,:,:,0,0])
        new_mask = np.expand_dims(np.expand_dims(self.crop_image(mask_im),-1),-1)

        input_image['mask'] = new_mask
        input_image[self.imagename] = np.concatenate(new_im_list,axis=4)

        return input_image, None