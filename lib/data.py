import os

import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import random
from lib.commons import *

import threading, queue

import concurrent.futures
import multiprocessing
import urllib.request
import time
import random
import itertools
import math
import tqdm

import nibabel as nib
import json

def vec2array(vector,mask):
    """
    vector (xyz,n)
    mask (x,y,z)
    """
    array = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2],vector.shape[-1]))
    array[mask>0,:] = vector
    return array
def array2vec(array,mask):
    """
    array (x,y,z,n)
    mask (x,y,z)
    """
    vector = array[mask>0,:]
    return vector
def mysortfun(e):
    return e['acq_time']
def time_diff_new(t1,t2):
    """
    t1,t2: e.x. 081012.515000 or 81012.515000
    return: t2-t1 in seconds 
    """
    import datetime
    if str(t1)[5] != '.':
        time_1 = datetime.timedelta(hours= int(str(t1)[0:2]) , minutes=int(str(t1)[2:4]), seconds=int(str(t1)[4:6]))
    else: 
        time_1 = datetime.timedelta(hours= int(str(t1)[0:1]) , minutes=int(str(t1)[1:3]), seconds=int(str(t1)[3:5]))
    if str(t2)[5] != '.':
        time_2 = datetime.timedelta(hours= int(str(t2)[0:2]) , minutes=int(str(t2)[2:4]), seconds=int(str(t2)[4:6]))
    else:
        time_2 = datetime.timedelta(hours= int(str(t2)[0:1]) , minutes=int(str(t2)[1:3]), seconds=int(str(t2)[3:5]))
    
    return datetime.timedelta.total_seconds(time_2 - time_1)
def time_diff(t1,t2):
    """
    t1,t2: e.x. 081012.515000
    return: t2-t1 in seconds 
    """
    import datetime
    time_1 = datetime.timedelta(hours= int(str(t1)[0:2]) , minutes=int(str(t1)[2:4]), seconds=int(str(t1)[4:6]))
    time_2 = datetime.timedelta(hours= int(str(t2)[0:2]) , minutes=int(str(t2)[2:4]), seconds=int(str(t2)[4:6]))
    return datetime.timedelta.total_seconds(time_2 - time_1)
def time_diff_json(t1,t2):
    """
    t1,t2: e.x. 08:10:12.515000
    return: t2-t1 in seconds 
    """
    import datetime
    if str(t1)[7]=='.':
        t1 = str(str(t1)[0:6]+'0'+str(t1)[6])
    if str(t2)[7]=='.':
        t2 = str(str(t2)[0:6]+'0'+str(t2)[6])
    time_1 = datetime.timedelta(hours= int(str(t1)[0:2]) , minutes=int(str(t1)[3:5]), seconds=int(str(t1)[6:8]))
    time_2 = datetime.timedelta(hours= int(str(t2)[0:2]) , minutes=int(str(t2)[3:5]), seconds=int(str(t2)[6:8]))
    return datetime.timedelta.total_seconds(time_2 - time_1)
def get_acq_time(jsonpath):
    _jsondata = open(jsonpath)

    jsondata=json.load(_jsondata)
    _jsondata.close()
    # deal with 17:31:2.585000
    old_acq = jsondata['AcquisitionTime']
    if str(old_acq)[7] == '.':
        old_acq = str(old_acq)[0:6]+'0'+str(old_acq)[6:]
    
    return jsondata['SeriesDescription'],old_acq

class MATLABConverter(object):
    # Transfomrations on U are different between batches
    U_TRANS_FORMAT_PILOT_PATIENTS = 0
    U_TRANS_FORMAT_VOLUNTEERS = 1
    U_TRANS_FORMAT_2ND_BATCH_PATIENTS = 2  # fftshift invovled

    def __init__(
            self,
            mat_filepath,
            output_root,
            u_trans_format=U_TRANS_FORMAT_PILOT_PATIENTS
    ):
        self.data_dict = {}
        self.mat_filepath = mat_filepath
        self.output_root = output_root
        self.u_trans_format = u_trans_format

    def write_data(self):
        import pickle as pkl
        import gzip
        out_file_name = os.path.dirname(self.mat_filepath).split(os.path.sep)[-1] \
                        + "_" + os.path.basename(self.mat_filepath).replace(".mat", ".pkl.gz")
        with gzip.GzipFile(
                os.path.join(
                    self.output_root,
                    out_file_name
                ), "wb") as f:
            pkl.dump(self.data_dict, f)

    def proc_data_dict(self):
        # Convert single item arrays
        self.data_dict['L'] = int(self.data_dict['L'])
        self.data_dict['Nx'], self.data_dict['Ny'], self.data_dict['Nz'] = int(self.data_dict['Nx']), int(
            self.data_dict['Ny']), int(
            self.data_dict['Nz'])
        self.data_dict['cw'] = float(self.data_dict['cw'])

        rev_U = self.data_dict['U'].reshape(self.data_dict['L'], self.data_dict['Nx'], self.data_dict['Nz'],
                                            self.data_dict['Ny'])
        rev_U = np.fft.fftshift(rev_U, axes=-1)
        rev_U = np.fft.fftshift(rev_U, axes=-2)
        rev_U = reverse_dimensions(rev_U)
        rev_U = np.flip(rev_U, axis=0)

        self.data_dict['U'] = rev_U

        return self.data_dict

    def run(self):
        # Convert
        self.data_dict = read_mat_file(self.mat_filepath)
        # Arrange the dataset for new batch of data
        self.data_dict = self.proc_data_dict()
        self.write_data()
        return 0


class ProcessorMTMRI(object):
    """ Synthesis DCE Reconstruction from Pickle File """
    FRAME_WINDOW = 215
    FRAME_BATCH = FRAME_WINDOW
    FULL_CHANNELS = False
    PATCH_SHAPE = (64, 64, 64)
    PATCH_PER_IMAGE = 16
    INJUSTION_FRAME = 78
    MAX_NUM_PHASE = 7
    NUM_PHASE = 4
    FOREGROUND_MASK_THRESHOLD = 0.02

    def __init__(
            self,
            offset_frames=None, full_channels=None,
            frame_window=None, frame_batch=None,
            patch_shape=None, patch_per_image=None,
            num_phase=None, max_num_phase=None,
            fg_thres=None, full_frame=None,
            device=None, randomness=True,
    ):

        # Configs

        self.offset_frames = self.INJUSTION_FRAME - 30 if offset_frames is None else offset_frames
        self.full_channels = self.FULL_CHANNELS if full_channels is None else full_channels
        self.FULL_FRAME = 215 if full_frame is None else full_frame
        if not self.full_channels:
            self.num_phase = self.NUM_PHASE if num_phase is None else num_phase
            self.frame_batch = self.FRAME_BATCH if frame_batch is None else frame_batch
            self.frame_window = self.FRAME_WINDOW if frame_window is None else frame_window
        else:
            # self.frame_window = np.random.randint(25, 48)  if frame_window is None else frame_window # corresponding to 70s to chao
            # self.frame_batch = self.frame_window
            # self.num_phase = np.minimum(self.FULL_FRAME // self.frame_window,
            #                             random.randint(4, 7 + 1))  if num_phase is None else num_phase # corresponding to 4 to 7 phases
            self.num_phase, self.frame_window = self.gen_rand_phase(self.FULL_FRAME, 25, 48, 4, 7)
            self.frame_batch = self.frame_window

        self.patch_shape = self.PATCH_SHAPE if patch_shape is None else patch_shape
        self.patch_per_image = self.PATCH_PER_IMAGE if patch_per_image is None else patch_per_image
        self.max_num_phase = self.MAX_NUM_PHASE if max_num_phase is None else max_num_phase
        self.fg_thres = self.FOREGROUND_MASK_THRESHOLD if fg_thres is None else fg_thres
        self.device = 'CPU:0' if device is None else device
        self.randomness = randomness

        # Data placeholder
        self.data_dict = dict()

        # Data Variables
        self.img_u = None
        self.img_l = None
        self.img_gr = None
        self.img_phi = None
        self.img_lm = None
        self.nx, self.ny, self.nz, self.cw = None, None, None, None

        self.weighted_vector = None

    def make_weighted_vector(self, frame_window=None):
        if frame_window is None:
            frame_window = self.frame_window
        if frame_window % 2 == 0:
            weighted_vector = np.concatenate(
                [np.linspace(0, 1.0, frame_window // 2 + 1)[1:], np.linspace(1.0, 0.0, frame_window // 2 + 1)[:-1]])
        else:
            weighted_vector = np.concatenate(
                [np.linspace(0, 1.0, frame_window // 2 + 2)[1:], np.linspace(1.0, 0.0, frame_window // 2 + 2)[1:-1]])
        weighted_vector = weighted_vector / sum(weighted_vector)
        return tf.cast(weighted_vector, tf.float32)

    def gen_rand_phase(self, full_frame, min_f, max_f, min_n, max_n):
        while 1:
            frame_window = np.random.randint(min_f, max_f)
            n_phase = np.minimum(full_frame // frame_window, random.randint(min_n, max_n + 1))
            if min_n <= n_phase <= max_n:
                return n_phase, frame_window

    @tf.function
    def tf_syn_dce(self, crp_u):
        start_phase_id = 0

        if self.randomness:
            rand_offset_frms = max(
                0,
                self.offset_frames + int(np.random.randint(-30 // 2, 30 // 2))
            )
        else:
            rand_offset_frms = 0

        with tf.device(self.device):
            __stacked_inputs = []
            __stacked_labels = []
            for phase_id in range(start_phase_id, start_phase_id + self.num_phase):
                __inp, __lbl = self.syn_dce_single_phase(
                    phase_id, rand_offset_frms, self.frame_batch, self.frame_window,
                    crp_u, self.img_gr, self.img_phi, self.img_l,
                    nx=self.patch_shape[0], ny=self.patch_shape[1], nz=self.patch_shape[2],
                    weighted_vector=self.weighted_vector
                )

                __inp = tf.cast(tnp.abs(__inp) / self.cw, tf.float32)
                __lbl = tf.cast(tnp.abs(__lbl) / self.cw, tf.float32)

                __stacked_inputs.append(__inp)
                __stacked_labels.append(__lbl)
            if self.full_channels:
                __stacked_inputs = tnp.concatenate(__stacked_inputs, axis=-1)
            else:
                __stacked_inputs = tnp.stack(__stacked_inputs, axis=-1)
            __stacked_labels = tnp.concatenate(__stacked_labels, axis=-1)

            # Normalization by mean and std
            # __inp_mean, __inp_std = tf.cast(tnp.mean(__stacked_labels),tf.float32), tf.cast(tnp.std(__stacked_labels),tf.float32)
            # __stacked_inputs = (__stacked_inputs - __inp_mean) / __inp_std
            # __stacked_labels = (__stacked_labels - __inp_mean) / __inp_std

            # Normalization by max and min 
            __inp_mean = tf.cast(tf.math.reduce_min(__stacked_labels), tf.float32)
            __inp_std = tf.cast(tf.math.reduce_max(__stacked_labels) - tf.math.reduce_min(__stacked_labels), tf.float32)
            __stacked_inputs = (__stacked_inputs - __inp_mean) / __inp_std
            __stacked_labels = (__stacked_labels - __inp_mean) / __inp_std

            if self.full_channels:
                __stacked_inputs = tnp.concatenate(__stacked_inputs, axis=-1)
            else:
                __stacked_inputs = tnp.stack(__stacked_inputs, axis=-1)
            __stacked_labels = tnp.concatenate(__stacked_labels, axis=-1)
        return __stacked_inputs, __stacked_labels, __inp_mean, __inp_std

    # weighted average over the phase window
    @tf.function
    def syn_dce_single_phase(self, phase_id, offset_frames, frame_batch, frame_window,
                             img_u, img_gr, img_phi, img_l,
                             nx, ny, nz,
                             weighted_vector
                             ):
        first_t_id = phase_id * frame_window + offset_frames
        last_t_id = (phase_id + 1) * frame_window + offset_frames

        # Try to use Tensorflow GPU
        avg_recon, all_recons = None, []

        for batch_id, st_t in enumerate(range(first_t_id, last_t_id, frame_batch)):
            t = list(range(st_t, st_t + frame_batch))
            if self.full_channels and phase_id == self.num_phase - 1:
                t = list(range(st_t, st_t + frame_batch + self.FULL_FRAME - frame_batch * self.num_phase))
            # Reconstruction frames
            recon = tf.cast(
                tnp.abs(
                    tnp.reshape(
                        tnp.matmul(
                            tnp.reshape(img_u, (-1, img_l)),
                            (
                                tf.linalg.solve(
                                    img_gr,
                                    tf.gather(img_phi[:, -1, :], t, axis=-1))
                            )
                        ),
                        (nx, ny, nz, len(t))
                    )
                ),
                tf.float32)
            if phase_id != 0: #consider post-contrast
                __avg_frame = tnp.sum(
                    recon[:, :, :, :frame_batch] * weighted_vector[st_t - first_t_id:st_t - first_t_id + frame_batch],
                    axis=-1)
            else: #pre-contrast 
                __avg_frame = tnp.reshape(recon[:, :, :, 0],(nx,ny,nz))
            if avg_recon is None:
                avg_recon = tnp.zeros_like(__avg_frame)

            avg_recon += __avg_frame
            if len(all_recons) == 0:
                all_recons = recon
            else:
                all_recons = tnp.concatenate([all_recons, recon], axis=-1)
            if self.full_channels:
                # print('current phase:',phase_id,self.num_phase-1)
                if phase_id < self.num_phase - 1:
                    # avg_recon = tnp.concatenate(
                    #     [tnp.zeros((nx,ny,nz,frame_batch//2)),
                    #      tf.reshape(avg_recon,(nx,ny,nz,-1)),
                    #      tnp.zeros((nx,ny,nz,(frame_batch-1)//2))],axis=-1)
                    avg_recon = tnp.repeat(tf.reshape(avg_recon, (nx, ny, nz, -1)), frame_batch, axis=-1)
                else:
                    # avg_recon = tnp.concatenate(
                    #     [tnp.zeros((nx,ny,nz,frame_batch//2)),
                    #      tf.reshape(avg_recon,(nx,ny,nz,-1)),
                    #      tnp.zeros((nx,ny,nz,(frame_batch-1)//2+self.FULL_FRAME-frame_batch*self.num_phase))],axis=-1)
                    avg_recon = tnp.repeat(tf.reshape(avg_recon, (nx, ny, nz, -1)),
                                           frame_batch + self.FULL_FRAME - frame_batch * self.num_phase, axis=-1)
        return avg_recon, all_recons

    # weighted average over the phase window
    @tf.function
    def recon_dce(self, offset_frames,
                  img_u, img_gr, img_phi, img_l,
                  nx, ny, nz,
                  ):
        first_t_id = offset_frames
        last_t_id = offset_frames + self.FULL_FRAME

        # Try to use Tensorflow GPU
        avg_recon, all_recons = None, []
        st_t = first_t_id
        t = list(range(first_t_id, last_t_id))
        # Reconstruction frames
        recon = tnp.abs(
            tnp.reshape(
                tnp.matmul(
                    tnp.reshape(img_u, (-1, img_l)),
                    (
                        tf.linalg.solve(
                            img_gr,
                            tf.gather(img_phi[:, -1, :], t, axis=-1))
                    )
                ),
                (nx, ny, nz, len(t))
            )
        )

        if len(all_recons) == 0:
            all_recons = recon
        else:
            all_recons = tnp.concatenate([all_recons, recon], axis=-1)

        return all_recons

    def random_cropping_batches(self, img_u, patch_shape=None, patch_per_image=None):
        # Create foreground area
        if patch_shape is None:
            patch_shape = self.patch_shape
        if patch_per_image is None:
            patch_per_image = self.patch_per_image

        fg_thres = self.fg_thres

        if fg_thres > 0:
            mean_rank_value = np.mean(np.abs(img_u), axis=-1)
            foreground_regions = mean_rank_value > fg_thres * np.max(mean_rank_value)
        else:
            foreground_regions = np.ones_like(np.mean(np.abs(img_u), axis=-1))

        # Randomly select foreground centers
        fg_x, fg_y, fg_z = np.where(foreground_regions)
        picked_indices = np.random.choice(len(fg_x), patch_per_image, replace=False)
        x, y, z = fg_x[picked_indices], fg_y[picked_indices], fg_z[picked_indices]

        # Add cropped batches to list
        img_u_patches = []
        for __x, __y, __z in zip(x, y, z):
            crp_u = crop_center_image_xyz(img_u, (__x, __y, __z), patch_shape)
            img_u_patches.append(crp_u)
        return img_u_patches

    def load_data(self, file_path):
        import gzip
        import pickle as pkl
        with gzip.GzipFile(file_path, "rb") as f:
            data_dict = pkl.load(f)
        self.data_dict = data_dict

        # Process
        self.img_u = np.transpose(data_dict['U'], (2, 0, 1, 3))  # YZX (MATLAB) ->  XYZ
        self.img_l = int(data_dict['L'])
        self.img_gr = data_dict['Gr']
        self.img_phi = np.squeeze(data_dict['Phi'])
        self.img_lm = data_dict['lm']
        self.nx, self.ny, self.nz, self.cw = \
            int(data_dict['Nx']), int(data_dict['Ny']), int(data_dict['Nz']), np.cast['float32'](
                data_dict['cw'])

        self.weighted_vector = self.make_weighted_vector(self.frame_window)

        return 0

    def get_image_data(self):
        # Preprocessing
        # Random cropping
        batch_crp_u = self.random_cropping_batches(
            self.img_u, self.patch_shape, self.patch_per_image
        )  # Cropping U
        inputs = []
        labels = []
        means = []
        stds = []
        # print('random:',self.num_phase,self.frame_window,self.offset_frames)
        # Synthesis Image
        for crp_u in batch_crp_u:
            u_inputs, u_labels, u_mean, u_std = self.tf_syn_dce(crp_u)

            inputs.append(u_inputs)
            labels.append(u_labels)
            means.append(u_mean)
            stds.append(u_std)
        return inputs, labels, means, stds

class ProcessorMTMRI_ROI(object):
    """
    Synthesis DCE Reconstruction from Pickle File
    With left ventricular (AIF) region, determine arterial phase and the subsequent phases
    get several (192,256,16) volumes
    Crop mask_tum mask_br and mask_bd as needed
    step1: get image data, patch U and mask into volumes
    step2: determine timing by AIF
    step3: reconstruct DCE according to the timing
    step4: return images and mask
    step5: inference by model
    step6: get image back using patcher
    step7: quantification
    step8: ROI analysis

    When using, set offset_frames=None,full_channels=True,
    frame_window=None,frame_batch=None,
    patch_shape optional,patch_per_image optional
    num_phase=None,max_num_phase=None,
    fg_thres optional, full_frame optional (recommend 180)
    """
    FRAME_WINDOW = 30
    FULL_CHANNELS = True
    PATCH_SHAPE = (256,192,16)
    PATCH_PER_IMAGE = 1
    INJECTION_FRAME = 70
    MAX_NUM_PHASE = 7
    NUM_PHASE = 4
    FOREGROUND_MASK_THRESHOLD = 0.02

    def __init__(
            self,
            offset_frames=None, full_channels=None,
            frame_window=None, frame_batch=None,
            patch_shape=None,
            num_phase=None, max_num_phase=None,
            fg_thres=None, full_frame=None,
            device=None, randomness=True,
    ):

        # Configs
        self.offset_frames = self.INJECTION_FRAME - 30 if offset_frames is None else offset_frames
        self.full_channels = self.FULL_CHANNELS if full_channels is None else full_channels
        self.FULL_FRAME = 180 if full_frame is None else full_frame
        self.num_phase, self.frame_window = None,None 
        self.frame_batch = self.frame_window
       
        self.patch_shape = self.PATCH_SHAPE if patch_shape is None else patch_shape
        # self.patch_per_image = self.PATCH_PER_IMAGE if patch_per_image is None else patch_per_image
        self.max_num_phase = self.MAX_NUM_PHASE if max_num_phase is None else max_num_phase
        self.fg_thres = self.FOREGROUND_MASK_THRESHOLD if fg_thres is None else fg_thres
        self.device = 'CPU:0' if device is None else device
        self.randomness = randomness

        # Data placeholder
        self.data_dict = dict()

        # Data Patcher
        self.image_patcher = None

        # Data Variables
        self.img_u = None
        self.img_l = None
        self.img_gr = None
        self.img_phi = None
        self.img_lm = None
        self.nx, self.ny, self.nz, self.cw = None, None, None, None

        self.weighted_vector = None
    def gen_rand_phase(self, full_frame, min_f, max_f, min_n, max_n):
        while 1:
            frame_window = np.random.randint(min_f, max_f)
            n_phase = np.minimum(full_frame // frame_window, random.randint(min_n, max_n + 1))
            if min_n <= n_phase <= max_n:
                return n_phase, frame_window
    def random_cropping_batches(self, img_u, patch_shape=None, patch_per_image=None):
        # Create foreground area
        if patch_shape is None:
            patch_shape = self.patch_shape

        patch_z = patch_per_image[-1]
        fg_thres = self.fg_thres

        if fg_thres > 0:
            mean_rank_value = np.mean(np.abs(img_u), axis=-1)
            foreground_regions = mean_rank_value > fg_thres * np.max(mean_rank_value)
        else:
            foreground_regions = np.ones_like(np.mean(np.abs(img_u), axis=-1))

        # Randomly select foreground centers
        fg_x, fg_y, fg_z = np.where(foreground_regions)
        picked_indices = np.random.choice(len(fg_x), 1, replace=False)
        x, y = fg_x[picked_indices], fg_y[picked_indices]
        z = range(patch_z,self.nz,patch_z)
        x, y = np.repeat(x,len(z)), np.repeat(y,len(z))

        # Add cropped batches to list
        img_u_patches = []
        for __x, __y, __z in zip(x, y, z):
            crp_u = crop_center_image_xyz(img_u, (__x, __y, __z), patch_shape)
            img_u_patches.append(crp_u)
        return img_u_patches
    def make_weighted_vector(self, frame_window=None):
        if frame_window is None:
            frame_window = self.frame_window
        if frame_window % 2 == 0:
            weighted_vector = np.concatenate(
                [np.linspace(0, 1.0, frame_window // 2 + 1)[1:], np.linspace(1.0, 0.0, frame_window // 2 + 1)[:-1]])
        else:
            weighted_vector = np.concatenate(
                [np.linspace(0, 1.0, frame_window // 2 + 2)[1:], np.linspace(1.0, 0.0, frame_window // 2 + 2)[1:-1]])
        weighted_vector = weighted_vector / sum(weighted_vector)
        return tf.cast(weighted_vector, tf.float32)

    def gen_rand_phase(self, full_frame, min_f, max_f, min_n, max_n):
        while 1:
            frame_window = np.random.randint(min_f, max_f)
            n_phase = np.minimum(full_frame // frame_window, random.randint(min_n, max_n + 1))
            if min_n <= n_phase <= max_n:
                return n_phase, frame_window
    def load_data(self, file_path):
        import gzip
        import pickle as pkl
        with gzip.GzipFile(file_path, "rb") as f:
            data_dict = pkl.load(f)
        self.data_dict = data_dict

        # Process
        self.img_u = np.transpose(data_dict['U'], (2, 0, 1, 3))  # YZX (MATLAB) ->  XYZ
        self.img_mask = np.transpose(data_dict['all_mask'], (2, 0, 1)) # YZX (MATLAB) ->  XYZ
        self.img_l = int(data_dict['L'])
        self.img_gr = data_dict['Gr']
        self.img_phi = np.squeeze(data_dict['Phi'])
        self.img_phi = np.concatenate((self.img_phi,np.repeat(self.img_phi[:,:,-1:],100,axis=-1)),axis=-1)
        self.img_lm = data_dict['lm']
        self.nx, self.ny, self.nz, self.cw = \
            int(data_dict['Nx']), int(data_dict['Ny']), int(data_dict['Nz']), np.cast['float32'](
                data_dict['cw'])

        
        return 0
    def calc_phase_timing(self):
        """
        calculate phase timing by AIF curve
        """
        # Reconstruction AIF
        mask_AIF = self.img_mask==3
        img_u = self.img_u[mask_AIF,] # get AIF
        img_phi, img_gr, img_l, nx, ny, nz = self.img_phi, self.img_gr, self.img_l, self.nx, self.ny, self.nz
        
        recon = tf.cast(
            tnp.abs(
                tnp.reshape(
                    tnp.matmul(
                        tnp.reshape(img_u, (-1, img_l)),
                        (
                            tf.linalg.solve(
                                img_gr,
                                img_phi[:, -1, :])
                        )
                    ),
                    (img_u.shape[0], -1)
                )
            ),
            tf.float32)
        

        # search max value and that is arterial phase
        recon = np.mean(recon,axis=0)
        recon = moving_average(recon)
        arterial_timing, total_len = np.argmax(recon), recon.shape[-1]
        self.num_phase, self.frame_window = \
            self.gen_rand_phase(total_len-100-arterial_timing-20, 25, 48, 4-1, 7-1) # ensure the last phase at least 20 steps long
        self.num_phase+=1 # add pre-contrast phase
        self.frame_batch = self.frame_window
        self.weighted_vector = self.make_weighted_vector(10) #10 phases for averaging 
        self.arterial = recon 

        total_timing = []
        pre_timing = arterial_timing-20 if arterial_timing>20 else 0 # should be able to find the pre-contrast phase
        total_timing.append(pre_timing)
        for i in range(self.num_phase-1):
            total_timing.append(i*self.frame_window+arterial_timing)
        total_timing.append(pre_timing+self.FULL_FRAME)
        self.total_timing = total_timing # note the timing is the timing in Phi; not the output
        print(total_timing)
        return 0

    @tf.function
    def tf_syn_dce(self, crp_u):
        """
        synthesize multi-phasic DCE according to the timing
        from pre_timing, count for self.full_frame

        :param crp_u:cropped U in [patch_size,L]
        :return:
        """
        total_timing = self.total_timing

        with tf.device(self.device):
            __stacked_inputs = []
            __stacked_labels = []
            for phase_id in range(self.num_phase):
                __inp, __lbl = self.syn_dce_single_phase(
                    phase_id, self.frame_batch, self.frame_window,
                    crp_u, self.img_gr, self.img_phi, self.img_l,
                    nx=self.patch_shape[0], ny=self.patch_shape[1], nz=self.patch_shape[2],
                    weighted_vector=self.weighted_vector
                )

                __inp = tf.cast(tnp.abs(__inp) / self.cw, tf.float32)
                __lbl = tf.cast(tnp.abs(__lbl) / self.cw, tf.float32)

                __stacked_inputs.append(__inp)
                __stacked_labels.append(__lbl)
            if self.full_channels:
                __stacked_inputs = tnp.concatenate(__stacked_inputs, axis=-1)
            else:
                __stacked_inputs = tnp.stack(__stacked_inputs, axis=-1)
            __stacked_labels = tnp.concatenate(__stacked_labels, axis=-1)

            # Normalization by mean and std
            # __inp_mean, __inp_std = tf.cast(tnp.mean(__stacked_labels),tf.float32), tf.cast(tnp.std(__stacked_labels),tf.float32)
            # __stacked_inputs = (__stacked_inputs - __inp_mean) / __inp_std
            # __stacked_labels = (__stacked_labels - __inp_mean) / __inp_std

            # Normalization by max and min
            __inp_mean = tf.cast(tf.math.reduce_min(__stacked_labels), tf.float32)
            __inp_std = tf.cast(tf.math.reduce_max(__stacked_labels) - tf.math.reduce_min(__stacked_labels), tf.float32)
            __stacked_inputs = (__stacked_inputs - __inp_mean) / __inp_std
            __stacked_labels = (__stacked_labels - __inp_mean) / __inp_std

            if self.full_channels:
                __stacked_inputs = tnp.concatenate(__stacked_inputs, axis=-1)
            else:
                __stacked_inputs = tnp.stack(__stacked_inputs, axis=-1)
            __stacked_labels = tnp.concatenate(__stacked_labels, axis=-1)
        return __stacked_inputs, __stacked_labels, __inp_mean, __inp_std

    # weighted average over the phase window
    @tf.function
    def syn_dce_single_phase(self, phase_id, frame_batch, frame_window,
                             img_u, img_gr, img_phi, img_l,
                             nx, ny, nz,
                             weighted_vector
                             ):
        """
        frame_window: how many frames are the synthesized phase representing 
        frame_batch: how many frames are the synthesized phase coming from 
        """
        print('phase synthesized %d in %d'%(phase_id,self.num_phase))
        first_t_id = self.total_timing[phase_id] 
        last_t_id = self.total_timing[phase_id+1]
        frame_batch = last_t_id-first_t_id
        frame_window = frame_batch 
        
        # Try to use Tensorflow GPU
        avg_recon, all_recons = None, []

        for batch_id, st_t in enumerate(range(first_t_id, last_t_id, frame_batch)):
            t = list(range(st_t, st_t + frame_batch)) 
            
            # Reconstruction frames
            recon = tf.cast(
                tnp.abs(
                    tnp.reshape(
                        tnp.matmul(
                            tnp.reshape(img_u, (-1, img_l)),
                            (
                                tf.linalg.solve(
                                    img_gr,
                                    tf.gather(img_phi[:, -1, :], t, axis=-1))
                            )
                        ),
                        (nx, ny, nz, len(t))
                    )
                ),
                tf.float32)
            if phase_id is not 0: #consider post-contrast
                print('f',frame_window,weighted_vector)
                __avg_frame = tnp.sum(
                    recon[:, :, :, frame_window//2-5:frame_window//2+5] * weighted_vector,
                    axis=-1)
            else: #pre-contrast
#                 __avg_frame = tnp.reshape(tnp.mean(recon[:, :, :, 0:5],axis=-1),(nx,ny,nz))
                __avg_frame = tnp.reshape(recon[:, :, :, 0],(nx,ny,nz))
            if avg_recon is None:
                avg_recon = tnp.zeros_like(__avg_frame)

            avg_recon += __avg_frame

            if self.full_channels:
                # print('current phase:',phase_id,self.num_phase-1)
                avg_recon = tnp.repeat(tf.reshape(avg_recon, (nx, ny, nz, -1)), frame_window, axis=-1)
                all_recons = recon 
#                 else:
#                     # avg_recon = tnp.concatenate(
#                     #     [tnp.zeros((nx,ny,nz,frame_batch//2)),
#                     #      tf.reshape(avg_recon,(nx,ny,nz,-1)),
#                     #      tnp.zeros((nx,ny,nz,(frame_batch-1)//2+self.FULL_FRAME-frame_batch*self.num_phase))],axis=-1)
#                     avg_recon = tnp.repeat(tf.reshape(avg_recon, (nx, ny, nz, -1)),
#                                            self.FULL_FRAME - (20+frame_window*(self.num_phase-2)), axis=-1)
#                     all_recons = recon
#                     print(avg_recon.shape,all_recons.shape)

        return avg_recon, all_recons
    def get_image_data(self):
        # Preprocessing
        # Random cropping
        """
            step1: get image data, patch U and mask into volumes
        """
        self.image_patcher = ImagePatcher(self.img_u,self.patch_shape)
        batch_crp_u = self.image_patcher.volume_to_patches()
        self.mask_patcher = ImagePatcher(self.img_mask,self.patch_shape)
        batch_crp_m = self.mask_patcher.volume_to_patches()

        """
            step2: determine timing by AIF
        """
        self.calc_phase_timing()

        inputs = []
        labels = []
        means = []
        stds = []
        masks = []

        """
            step3: reconstruct DCE according to the timing
        """

        # Synthesis Image
        for count, crp_u in enumerate(batch_crp_u):
            u_inputs, u_labels, u_mean, u_std = self.tf_syn_dce(crp_u)

            inputs.append(u_inputs)
            labels.append(u_labels)
            means.append(u_mean)
            stds.append(u_std)
            masks.append(batch_crp_m[count])

        """
        step4: return images and mask
        """
        self.mask_patcher = []
        self.image_patcher = []
    
        return inputs, labels, means, stds, masks

class MultiThreadedDataGenerator(object):
    def __init__(
            self,
            iter_inp,
            n_workers=32, n_max_loader=None, batch_size=4, iter_per_epoch=None, q_length=None,
            shuffle=True, min_after_dequeue=None, shuffle_each_batch=False, frame_window=None,
            patch_per_image=16, num_phase=4, patch_shape=(64, 64, 64), q_max_wait=None,
            full_channels=False, debug=False):
        """
        iter_inp:   Iterable

        """
        # inputs
        self.iter_inp = iter_inp
        self.iter_inp_index = 0  # Index position for updating the futures with new input item

        # configs
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.q_max_wait = q_max_wait

        self.n_max_loader = max(min(n_workers // 4 if n_max_loader is None else n_max_loader, n_workers), 1)
        self.patch_per_image = patch_per_image
        self.num_phase = num_phase
        self.patch_shape = patch_shape
        self.iter_per_epoch = math.ceil(
            (len(iter_inp) * patch_per_image) / self.batch_size) if iter_per_epoch is None else iter_per_epoch
        self.frame_window = frame_window,
        # Shuffle Relevant
        self.shuffle = shuffle
        self.q_length = 8 * batch_size if q_length is None else q_length
        self.min_after_dequeue = min(4 * batch_size if min_after_dequeue is None else min_after_dequeue, self.q_length)
        self.shuffle_each_batch = shuffle_each_batch
        self.num_of_shuffled_item = 0  # How many items in the queue has been shuffle

        self.current_batch_id = 0
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.n_current_loader = 0
        self.data_q = queue.Queue(self.q_length)
        self.executor = None
        self.futures = []
        self.debug = debug

    def constrainted_data_loader(self, __inp, m_preproc):
        # Check available loading slots
        while True:
            with self.lock:
                if self.n_current_loader < self.n_max_loader:
                    self.n_current_loader += 1
                    break
            time.sleep(1)  # Wait until a slot for the loader

        m_preproc.load_data(__inp)

        with self.lock:
            self.n_current_loader -= 1

        return m_preproc

    def map_fn(self, __inp):
        """
        ### Replace the actual process function here ###
        """
        m_preproc = ProcessorMTMRI(
            offset_frames=None,
            frame_window=self.frame_window, frame_batch=self.frame_window,
            patch_shape=self.patch_shape, patch_per_image=self.patch_per_image,
            num_phase=self.num_phase, max_num_phase=None, full_channels=full_channels,
        )

        m_preproc = self.constrainted_data_loader(__inp, m_preproc)

        inps, lbls = m_preproc.get_image_data()

        return inps, lbls

    def enqueue(self, __inp):
        inps, lbls = self.map_fn(__inp)

        for __inp, __lbl in zip(inps, lbls):
            self.data_q.put((__inp, __lbl))

    def shuffle_queue(self):
        cached_list = []

        # Lock the queue
        with self.lock:
            num_shuffled_item = self.data_q.qsize()
            # Read the queue
            for _ in range(num_shuffled_item):
                cached_list.append(self.data_q.get())
                self.data_q.task_done()

            random.shuffle(cached_list)

            # Load to the queue
            for _ in cached_list:
                self.data_q.put(_)

        self.num_of_shuffled_item = num_shuffled_item

    def gen(self):
        """
            Generator function to call
        """
        self.start_queue()
        while True:
            # Check & Update Running Queue
            self.check_update_futures()
            if self.debug:
                print('check finished ')
            # Stopping Criteria
            if self.current_batch_id >= self.iter_per_epoch:
                self.current_batch_id = 0
                return
            if self.debug:
                print('pass stop crit')
            # Shuffle
            if self.shuffle and self.min_after_dequeue > 0:
                if self.shuffle_each_batch or self.num_of_shuffled_item <= 0:
                    self.shuffle_queue()

            # Batching
            batch_inp, batch_lbl = self.batch()
            if self.debug:
                print('next step:', batch_inp.shape, batch_lbl.shape)
            self.current_batch_id += 1
            if batch_inp is None or batch_lbl is None:
                if self.debug:
                    print('while true end')
                return
            else:
                if self.debug:
                    print('while true no end')
                yield batch_inp, batch_lbl

    def dequeue(self):
        item = self.data_q.get(True, self.q_max_wait)
        self.num_of_shuffled_item -= 1
        self.data_q.task_done()
        if self.debug:
            print('dequeue finished:')
        return item

    def batch(self):
        batch_inp = []
        batch_lbl = []
        for _ in range(self.batch_size):
            inp, lbl = self.dequeue()
            if inp != None and lbl != None:
                batch_inp.append(inp)
                batch_lbl.append(lbl)
                if self.debug:
                    print('valid input')
            else:
                if self.debug:
                    print('none input')
                break
        if len(batch_inp) > 0 and len(batch_lbl) == len(batch_inp):
            batch_inp = np.stack(batch_inp, axis=0)
            batch_lbl = np.stack(batch_lbl, axis=0)
            return batch_inp, batch_lbl
        else:
            return None, None

    def start_queue(self):
        # Start the queue if not done yet
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers)

        if len(self.futures) == 0:
            self.futures = [self.executor.submit(self.enqueue, __inp) for __inp in self.iter_inp]

    def check_update_futures(self):
        # Get finished futures
        done_indices = np.where([f.done() for f in self.futures])[0]

        # Update done futures with new future
        for done_index in done_indices:
            __inp = self.iter_inp[self.iter_inp_index]
            self.iter_inp_index = self.iter_inp_index + 1 if self.iter_inp_index < len(self.iter_inp) - 1 else 0
            self.futures[done_index] = self.executor.submit(self.enqueue, __inp)
        if self.debug:
            print('check future finish,lenth:', len(done_indices))
        return len(done_indices)

    def close(self, wait=False):
        """ Shutdown and clear the queue"""
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=wait)
            except Exception as exc:
                print(exc)
        for f in self.futures:
            f.cancel()
        while not self.data_q.empty():
            self.data_q.get()


class ImagePatcher(object):
    """
        Helper class to split a big 3D image into patches and merge back into a big image
    """

    def __init__(self, in_img, patch_shape, overlap=1):
        """

        :param in_img:              Input Image
        :param patch_shape:         Patch dimension
        :param overlap:             Overlap ratio
        """
        self.in_img = in_img
        self.in_img_shape = self.in_img.shape
        self.patch_shape = patch_shape
        self.overlap = max(1, overlap)

        self.batches = None
        self.target_shape = None
        self.pad_shape = None

        self.compute_batch_dimensions()

    def compute_batch_dimensions(self):
        if len(self.in_img_shape) == len(self.patch_shape):
            self.in_img = np.expand_dims(self.in_img,axis=-1)
            self.in_img_shape = self.in_img_shape+(1,)
        batches = np.ceil(np.divide(self.in_img.shape, self.patch_shape + (1,))).astype(int)[:-1]  # ignore channel
        target_shape = np.multiply(
            self.patch_shape,
            batches
        )

        self.batches = batches
        self.target_shape = target_shape

        return batches, target_shape

    def pad_to_target_shape(self):
        in_img = self.in_img
        target_shape = self.target_shape

        pad_shape = np.subtract(target_shape, in_img.shape[:-1]) // 2
        pad_shape = tuple((_, _) for _ in pad_shape)

        self.pad_shape = pad_shape
        return np.pad(
            in_img,
            pad_shape + ((0, 0),),
            mode='reflect'
        )

    def crop_to_inimg_shape(self, padded_img):
        if self.pad_shape is None:
            return padded_img
        slices = []
        for __pad in self.pad_shape:
            slices.append(slice(__pad[0], -__pad[1] if __pad[1] > 0 else None))
        slices.append(slice(None, None))
        return padded_img[
            tuple(slices)
        ]

    def volume_to_patches(self):
        in_img = self.in_img
        target_shape = self.target_shape
        batches = self.batches
        patch_shape = self.patch_shape

        padded_img = self.pad_to_target_shape()

        patches = []
        for __x in range(batches[0]):
            for __y in range(batches[1]):
                for __z in range(batches[2]):
                    patches.append(padded_img[
                                   patch_shape[0] * __x:(__x + 1) * patch_shape[0],
                                   patch_shape[1] * __y:(__y + 1) * patch_shape[1],
                                   patch_shape[2] * __z:(__z + 1) * patch_shape[2],
                                   ])
        return patches

    def patches_to_volume(self, patches):
        target_shape = self.target_shape
        patch_shape = self.patch_shape
        batches = self.batches
        patch_idx = 0
        if len(patches) == 0:
            return

        out_img = np.zeros(tuple(self.target_shape) + (patches[0].shape[-1],))
        for __x in range(batches[0]):
            for __y in range(batches[1]):
                for __z in range(batches[2]):
                    out_img[
                    patch_shape[0] * __x:(__x + 1) * patch_shape[0],
                    patch_shape[1] * __y:(__y + 1) * patch_shape[1],
                    patch_shape[2] * __z:(__z + 1) * patch_shape[2]
                    ] = patches[patch_idx]
                    patch_idx += 1
        return out_img

    def merge_patches(self, patches):
        out_img = self.patches_to_volume(patches)
        return self.crop_to_inimg_shape(out_img)





