# -*- coding: utf-8 -*-
"""
Created on Feb 6th 2018
@author: Ben A Duffy - University of Southern California

"""

from __future__ import absolute_import, print_function
import numpy as np
import math
from niftynet.layer.base_layer import Layer
from functools import wraps
from time import time

PRINT_TIMING = False

try:
    import finufftpy
    finufft = True
except ImportError:
    finufft = False


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if PRINT_TIMING:
            print('func:%r, took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap


def create_rotation_matrix_3d(angles):
    """
    given a list of 3 angles, create a 3x3 rotation matrix that describes rotation about the origin
    :param angles (list or numpy array) : rotation angles in 3 dimensions
    :return (numpy array) : rotation matrix 3x3
    """

    mat1 = np.array([[1., 0., 0.],
                     [0., math.cos(angles[0]), math.sin(angles[0])],
                     [0., -math.sin(angles[0]), math.cos(angles[0])]],
                    dtype='float')

    mat2 = np.array([[math.cos(angles[1]), 0., -math.sin(angles[1])],
                     [0., 1., 0.],
                     [math.sin(angles[1]), 0., math.cos(angles[1])]],
                    dtype='float')

    mat3 = np.array([[math.cos(angles[2]), math.sin(angles[2]), 0.],
                     [-math.sin(angles[2]), math.cos(angles[2]), 0.],
                     [0., 0., 1.]],
                    dtype='float')

    mat = (mat1 @ mat2) @ mat3
    return mat


class MotionSimLayer(Layer):
    """
    given a real valued 3D MRI image, simulate random translations and rotations

    """

    def __init__(self, image_name, std_rotation_angle=0, std_translation=10,
                 corrupt_pct=(15,20), freq_encoding_dim=(0,1,2), preserve_center_pct=0.07,
                 name='motion_sim',apply_mask=True, nufft=False, proc_scale=-1, num_pieces=8):

        """
        :param image_name (str): key in data dictionary
        :param std_rotation_angle (float) : std of rotations
        :param std_translation (float) : std of translations
        :param corrupt_pct (list of ints): range of percents
        :param freq_encoding_dim (list of ints): randomly choose freq encoding dim
        :param preserve_center_pct (float): percentage of k-space center to preserve
        :param name (str) : name of layer
        :param apply_mask (bool): apply mask to output or not
        :param nufft (bool): whether to use nufft for introducing rotations
        :param proc_scale (float or int) : -1 = piecewise, -2 = uncorrelated, or float for random walk scale
        :param num_pieces (int): number of pieces for piecewise constant simulation

       raises ImportError if nufft is true but finufft cannot be imported

        """

        super(MotionSimLayer, self).__init__(name=name)
        self.image_name = image_name
        self.trajectory = None
        self.preserve_center_frequency_pct = preserve_center_pct
        self.freq_encoding_choice = freq_encoding_dim
        self.frequency_encoding_dim = None
        self.proc_scale = proc_scale
        self.num_pieces = num_pieces
        self.std_rotation_angle, self.std_translation = std_rotation_angle, std_translation
        self.corrupt_pct_range = corrupt_pct
        self.apply_mask = apply_mask

        if self.proc_scale == -1:
            self._simulate_trajectory = self._piecewise_simulation
            print('using piecewise motion simulation')
        elif self.proc_scale == -2:
            self._simulate_trajectory = self._gaussian_simulation
            print('using uncorrelated gaussian simulation')
        elif self.proc_scale > 0:
            self._simulate_trajectory = self._random_walk_simulation
            print('using random walk')
        else:
            raise ValueError('invalid proc_scale: should be either -1,-2 or positive real valued')

        self.nufft = nufft
        if (not finufft) and nufft:
            raise ImportError('finufftpy cannot be imported')

    def _calc_dimensions(self, im_shape):
        """
        calculate dimensions based on im_shape
        :param im_shape (list/tuple) : image shape

        - sets self.phase_encoding_dims, self.phase_encoding_shape, self.num_phase_encoding_steps, self.frequency_encoding_dim
        - initializes self.translations and self.rotations

        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(self.frequency_encoding_dim)
        self.phase_encoding_dims = pe_dims
        im_shape = list(im_shape)
        self.im_shape = im_shape.copy()
        im_shape.pop(self.frequency_encoding_dim)
        self.phase_encoding_shape = im_shape
        self.num_phase_encoding_steps = self.phase_encoding_shape[0] * self.phase_encoding_shape[1]
        self.translations = np.zeros(shape=(3, self.num_phase_encoding_steps))
        self.rotations = np.zeros(shape=(3, self.num_phase_encoding_steps))
        self.frequency_encoding_dim = len(self.im_shape)-1 if self.frequency_encoding_dim == -1 else self.frequency_encoding_dim

    @staticmethod
    def random_walk_trajectory(length,start_scale=10,proc_scale=0.1):
        seq = np.zeros([3, length])
        seq[:, 0] = np.random.normal(loc=0.0, scale=start_scale, size=(3,))
        for i in range(length - 1):
            seq[:, i + 1] = seq[:, i] + np.random.normal(scale=proc_scale, size=(3,))
        return seq

    @staticmethod
    def piecewise_trajectory(length,n_pieces=4,scale_trans=10,scale_rot=3):
        """
        generate random piecewise constant trajectory with n_pieces
        :param length (int): length of trajectory
        :param n_pieces (int): number of pieces
        :param scale_trans (float): scale of normal distribution for translations
        :param scale_rot (float): scale of normal distribution for rotations
        :return: list of numpy arrays of size (3 x length) for translations and rotations
        """
        seq_trans = np.zeros([3, length])
        seq_rot = np.zeros([3, length])
        ind_to_split = np.random.choice(length,size=n_pieces)
        split_trans = np.array_split(seq_trans,ind_to_split,axis=1)
        split_rot = np.array_split(seq_rot,ind_to_split,axis=1)
        for i,sp in enumerate(split_trans):
            sp[:] = np.random.normal(scale=scale_trans, size=(3, 1))
        for i, sp in enumerate(split_rot):
            sp[:] = np.random.normal(scale=scale_rot, size=(3, 1))
        return seq_trans,seq_rot

    def _random_walk_simulation(self,length):
        rand_translations = self.random_walk_trajectory(length,
                                                        start_scale=self.std_translation,
                                                        proc_scale=self.proc_scale)
        rand_rotations = self.random_walk_trajectory(length,
                                                     start_scale=self.std_rotation_angle,
                                                     proc_scale=self.std_rotation_angle / 1000)
        return rand_translations,rand_rotations

    def _piecewise_simulation(self,length):
        num_pieces = np.random.choice(np.arange(1, self.num_pieces))
        rand_translations, rand_rotations = self.piecewise_trajectory(length, n_pieces=num_pieces,
                                                                      scale_trans=self.std_translation,
                                                                      scale_rot=self.std_rotation_angle)
        return rand_translations,rand_rotations

    def _gaussian_simulation(self,length):
        rand_translations = np.random.normal(size=[3, length], scale=self.std_translation)
        rand_rotations = np.random.normal(size=[3, length], scale=self.std_rotation_angle)
        return rand_translations,rand_rotations

    def _center_k_indices_to_preserve(self):
        """get center k indices of freq domain"""
        mid_pts = [int(math.ceil(x/2)) for x in self.phase_encoding_shape]
        num_pts_preserve = [math.ceil(self.preserve_center_frequency_pct*x) for x in self.phase_encoding_shape]
        ind_to_remove = {val+1:slice(mid_pts[i] - num_pts_preserve[i],mid_pts[i] + num_pts_preserve[i])
                         for i,val in enumerate(self.phase_encoding_dims)}
        ix_to_remove = [ind_to_remove.get(dim, slice(None)) for dim in range(4)]
        return ix_to_remove

    @timing
    def _simulate_random_trajectory(self):
        """
        simulates random trajectory using a random number of lines generated from corrupt_pct_range

        modifies self.translations and self.rotations

        """

        # Each voxel has a random translation and rotation for 3 dimensions.
        rand_translations_vox = np.zeros([3] + self.im_shape)
        rand_rotations_vox = np.zeros([3] + self.im_shape)

        # randomly choose PE lines to corrupt
        choose_from_list = [np.arange(i) for i in self.phase_encoding_shape]
        num_lines = [int(x/100*np.prod(self.phase_encoding_shape)) for x in self.corrupt_pct_range]

        # handle deterministic case where no range is given
        if num_lines[0] == num_lines[1]:
            num_lines = num_lines[0]
        else:
            num_lines = np.random.randint(num_lines[0],num_lines[1],size=1)

        if num_lines == 0:
            # allow no lines to be modified
            self.translations = rand_translations_vox.reshape(3, -1)
            self.rotations = rand_rotations_vox.reshape(3, -1)
            return

        motion_lines = []
        for i in range(len(self.phase_encoding_shape)):
            motion_lines.append(np.random.choice(choose_from_list[i], size=num_lines, replace=True).tolist())

        # sort by either first or second PE dim
        dim_to_sort_by = np.random.choice([0,1])
        motion_lines_sorted = [list(x) for x in zip(*sorted(zip(motion_lines[0], motion_lines[1]), key=lambda x:x[dim_to_sort_by]))]
        motion_lines = motion_lines_sorted

        # generate random motion parameters
        rand_translations,rand_rotations = self._simulate_trajectory(len(motion_lines[0]))

        # create indexing tuple ix
        motion_ind_dict = {self.phase_encoding_dims[i]:val for i,val in enumerate(motion_lines)}
        ix = [motion_ind_dict.get(dim, slice(None)) for dim in range(3)]
        ix = tuple(ix)

        # expand in freq-encoding dim
        new_dims = [3,rand_translations.shape[-1]]
        self.rand_translations = np.expand_dims(rand_translations,-1)
        self.rand_rotations = np.expand_dims(rand_rotations,-1)
        new_dims.append(self.im_shape[self.frequency_encoding_dim])
        self.rand_translations = np.broadcast_to(self.rand_translations,new_dims)
        self.rand_rotations = np.broadcast_to(self.rand_rotations,new_dims)

        #insert into voxel-wise motion parameters
        for i in range(3):
            rand_rotations_vox[(i,) + ix] = self.rand_rotations[i, :, :]
            rand_translations_vox[(i,) + ix] = self.rand_translations[i, :, :]

        ix_to_remove = self._center_k_indices_to_preserve()
        rand_translations_vox[ix_to_remove] = 0
        rand_rotations_vox[ix_to_remove] = 0

        self.translations = rand_translations_vox.reshape(3, -1)
        rand_rotations_vox = rand_rotations_vox.reshape(3, -1)
        self.rotations = rand_rotations_vox * (math.pi / 180.)  # convert to radians


    def gen_test_trajectory(self, translation,rotation):
        """
        # for testing - apply the same transformation at all Fourier (time) points
        :param translation (list/array of length 3):
        :param rotation (list/array of length 3):

        modifies self.translations, self.rotations in place
        """
        num_pts = np.prod(self.im_shape)
        self.translations = np.array([np.ones([num_pts, ]).flatten() * translation[0],
                                      np.ones([num_pts, ]).flatten() * translation[1],
                                      np.ones([num_pts, ]).flatten() * translation[2]])

        self.rotations = np.array([np.ones([num_pts, ]).flatten() * rotation[0],
                                   np.ones([num_pts, ]).flatten() * rotation[1],
                                   np.ones([num_pts, ]).flatten() * rotation[2]])

    @timing
    def _fft_im(self, image):
        output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)

        return output

    @timing
    def _ifft_im(self, freq_domain):
        output = np.fft.ifftshift(np.fft.ifftn(freq_domain))
        return output

    @timing
    def _translate_freq_domain(self, freq_domain):
        """
        image domain translation by adding phase shifts in frequency domain
        :param freq_domain - frequency domain data 3d numpy array:
        :return frequency domain array with phase shifts added according to self.translations:
        """

        lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape]
        meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
        grid_coords = np.array([mg.flatten() for mg in meshgrids])

        phase_shift = np.multiply(grid_coords, self.translations).sum(axis=0)  # phase shift is added
        exp_phase_shift = np.exp(-2j * math.pi * phase_shift)
        freq_domain_translated = np.multiply(exp_phase_shift, freq_domain.flatten(order='C')).reshape(freq_domain.shape)

        return freq_domain_translated

    @timing
    def _rotate_coordinates(self):
        """

        :return: grid_coordinates after applying self.rotations

        """
        center = [math.ceil((x - 1) / 2) for x in self.im_shape]

        [i1, i2, i3] = np.meshgrid(np.arange(self.im_shape[0]) - center[0],
                                   np.arange(self.im_shape[1]) - center[1],
                                   np.arange(self.im_shape[2]) - center[2], indexing='ij')

        grid_coordinates = np.array([i1.T.flatten(), i2.T.flatten(), i3.T.flatten()])

        rotations = self.rotations.reshape([3] + self.im_shape)
        ix = (len(self.im_shape) + 1) * [slice(None)]
        ix[self.frequency_encoding_dim + 1] = 0  # dont need to rotate along freq encoding

        rotations = rotations[ix].reshape(3,-1)
        rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d,axis=0,arr=rotations).transpose([-1,0,1])
        rotation_matrices = rotation_matrices.reshape(self.phase_encoding_shape + [3,3])
        rotation_matrices = np.expand_dims(rotation_matrices, self.frequency_encoding_dim)

        rotation_matrices = np.tile(rotation_matrices,
                                    reps=([self.im_shape[self.frequency_encoding_dim] if i == self.frequency_encoding_dim else 1 for i in range(5)]))  # tile in freq encoding dimension

        rotation_matrices = rotation_matrices.reshape([-1, 3, 3])

        # tile grid coordinates for vectorizing computation
        grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
        grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
        rotation_matrices = rotation_matrices.reshape([-1, 3])

        new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)

        # reshape new grid coords back to 3 x nvoxels
        new_grid_coords = new_grid_coords.reshape([3, -1], order='F')

        # scale data between -pi and pi
        max_vals = [abs(x) for x in grid_coordinates[:, 0]]
        new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in
                                       range(new_grid_coords.shape[0])]
        new_grid_coordinates_scaled = [np.asfortranarray(i) for i in new_grid_coordinates_scaled]

        return new_grid_coordinates_scaled,[grid_coordinates,new_grid_coords]

    @timing
    def _nufft(self, freq_domain_data, iflag=1, eps=1E-7):
        """
        rotate coordinates and perform nufft
        :param freq_domain_data:
        :param iflag/eps: see finufftpy doc
        :param eps: precision of nufft
        :return: nufft of freq_domain_data after applying self.rotations
        """

        if not finufft:
            raise ImportError('finufftpy not available')

        new_grid_coords = self._rotate_coordinates()[0]

        # initialize array for nufft output
        f = np.zeros([len(new_grid_coords[0])], dtype=np.complex128, order='F')

        freq_domain_data_flat = np.asfortranarray(freq_domain_data.flatten(order='F'))

        finufftpy.nufft3d1(new_grid_coords[0], new_grid_coords[1], new_grid_coords[2], freq_domain_data_flat,
                            iflag, eps, self.im_shape[0], self.im_shape[1],
                            self.im_shape[2], f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                            chkbnds=0, upsampfac=1.25) # upsampling at 1.25 saves time at low precisions

        im_out = f.reshape(self.im_shape, order='F')

        return im_out

    @timing
    def layer_op(self, input_data, mask=None):

        self.frequency_encoding_dim = np.random.choice(self.freq_encoding_choice)

        image_data = input_data[self.image_name]

        original_image = np.squeeze(image_data[:,:,:,0,0])
        self._calc_dimensions(original_image.shape)
        self._simulate_random_trajectory()

        # fft
        im_freq_domain = self._fft_im(original_image)
        translated_im_freq_domain = self._translate_freq_domain(freq_domain=im_freq_domain)

        # iNufft for rotations
        if self.nufft:
            corrupted_im = self._nufft(translated_im_freq_domain)
            corrupted_im = corrupted_im / corrupted_im.size # normalize

        else:
            corrupted_im = self._ifft_im(translated_im_freq_domain)

        # magnitude
        corrupted_im = abs(corrupted_im)

        if self.apply_mask:
            # todo: use input arg mask
            mask_im = input_data['mask'][:,:,:,0,0]>0
            corrupted_im = np.multiply(corrupted_im, mask_im)
            masked_original = np.multiply(original_image, mask_im)
            image_data[:,:,:,0,0] = masked_original

        image_data[:,:,:,0,1] = corrupted_im
        
        output_data = input_data
        output_data[self.image_name] = image_data
        
        return output_data, mask
