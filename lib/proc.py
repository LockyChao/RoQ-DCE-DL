"""
signal processing functions
"""
import numpy as np
from scipy.signal import butter,filtfilt
import h5py
from lib.data import *
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def search_rise_tp(con_bd):
    """
    Goal: To find injection time point given concentration curve (preferable blood)
    Input: concentration curve or R1 curve
    Output:
    - R1_0: base R1 of blood
    - ind: index when R1 start to rise
    Algo: To find maximal 2nd derivative 
    """
    con_lowpass = butter_lowpass_filter(con_bd,cutoff = 2, fs = 30.0, order = 2)
    deriv = np.diff(con_lowpass,2)
    ind = np.minimum(np.argmax(deriv),10)
#     ind = np.where(deriv.flatten()>0.01)[0][10]
#     print('testing')
#     R1_0 = con[:,ind]
    return ind
def search_satu_tp(con_bd):
    """
    Goal: To find end-injection time point given concentration curve (preferable blood)
    Input: concentration curve or R1 curve
    Output:
    - R1_d: end-injection R1 of blood
    - ind: index when R1 start to drop
    Algo: To find maximal concentration
    """
    con_lowpass = butter_lowpass_filter(con_bd,cutoff = 2, fs = 30.0, order = 2)
    ind = np.argmax(con_lowpass)
    return ind
    
def gen_inv_con(R1s,r=3.2,ind_0=None):
    # calculate concentration from dynamic R1
    if ind_0 == None:
        ind_0 = search_rise_tp(R1s)
    con = (R1s - R1s[ind_0])/r
    return con
# def gen_sep_con(con,ind_0=None,ind_1=None):
#     """
#     Goal: To seperate concentration curve to pre-contrast, injection, and after injection
#     """
#     if ind_0 == None:
#         ind_0 = search_rise_tp(con)
#     if ind_1 == None:
#         ind_1 = search_satu_tp(con)
#     length = len(con)
#     c1 = con[0:int(ind_0)]
#     c2 = con[int(ind_0):int(ind_1)]
#     c3 = con[int(ind_1):length]
#     return c1,c2,c3

# ---------- define tool ...
def shuffle_x_y(array1,array2):
    shuffler = np.random.permutation(np.shape(array1)[0])
    array1_shuffled = array1[shuffler]
    array2_shuffled = array2[shuffler]
    return array1_shuffled,array2_shuffled
def shuffle_x_y_z(array1,array2,array3):
    shuffler = np.random.permutation(np.shape(array1)[0])
    array1_shuffled = array1[shuffler]
    array2_shuffled = array2[shuffler]
    array3_shuffled = array3[shuffler]
    return array1_shuffled,array2_shuffled,array3_shuffled
def shuffle_x_y_z_u(array1,array2,array3,array4):
    shuffler = np.random.permutation(np.shape(array1)[0])
    array1_shuffled = array1[shuffler]
    array2_shuffled = array2[shuffler]
    array3_shuffled = array3[shuffler]
    array4_shuffled = array4[shuffler]
    return array1_shuffled,array2_shuffled,array3_shuffled,array4_shuffled

def gen_norm_dist(size,low_tem=70//2.8,high_tem=140//2.8):
    import numpy as np
    from scipy.stats import truncnorm
    import matplotlib.pyplot as plt

    _scale = (high_tem-low_tem)//6
    _range = (high_tem-low_tem)//2
    
    X = truncnorm(a=-_range, b=+_range, scale=_scale,loc=(low_tem+high_tem)/2).rvs(size=size)
    X = X.round().astype(int)
    return X

def gen_inge_R1_known_tres_tps(R1_ori,R1_bd_ori,t_res,t_ps,ori_len):
    s_time=20
    index = []
    index.append(0)
    for inds in range(t_ps-1):
        index.append(s_time+inds*t_res)
    index = np.array(index)
    
    n_num = R1_ori.shape[0]
    R1 = gen_inge_spike(R1_ori,(index,t_res,ori_len))
    return R1

def gen_inge_R1_ext_random(R1,sample_num,R1_bd,sample_points=None,sample_pattern=None,augmentation=False,rise_index=None,OUT_DIM=None,average=False,debug=False,knownbundle=None,temp_res=2.8):
    """
    input shape: (batch,OUT_DIM)
    output shape: (batch,sample_num)
    """    
    R1 = np.concatenate((np.repeat(R1[:,0:1],40,axis=1),R1,np.repeat(np.reshape(R1[:,-1],(-1,1)),60,axis=1)),axis=-1)
    R1_bd = np.concatenate((np.repeat(R1_bd[:,0:1],40,axis=1),R1_bd,np.repeat(np.reshape(R1_bd[:,-1],(-1,1)),60,axis=1)),axis=-1)


    if augmentation == False:
        s_time = 30
    else:
#         s_time = np.random.randint(40//2,62//2,1)[0]
#         s_time = np.random.randint(20//2, 40//2, 1)[0]
        s_time = np.random.randint(15,25,1)[0]
    n_ts = R1.shape[0]        
    
    arterial_index = np.argmax(R1_bd)
    R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
    R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
    R1_itnpl = np.zeros_like(R1)
    R1_bd_itnpl = np.zeros_like(R1)
    nums = gen_norm_dist(n_ts)
    bundle = []
    pre_index = 0
    R1[:,pre_index] = 0
    for i in range(n_ts):
        
        _R1 = R1[i,:]
        _R1_bd = R1_bd[0,:]
        ori_len = R1.shape[-1]
        ds_post = nums[i]
#         print(ds_post)
        sample_num = ((OUT_DIM-s_time-1)//ds_post)+2
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index

        index = []
        index.append(pre_index)
        for inds in range(sample_num-1):
            index.append(s_time+inds*ds_post)
            
        index = np.array(index)

        
        _to_return = _R1[index]
#         return _to_return,R1,(index,ds,ori_len),R1_bd[:,index],R1_bd
        R1_itnpl[i,:] = gen_inge_spike(_to_return,(index,ds,ori_len))
        R1_bd_itnpl[i,:] = gen_inge_spike(_R1_bd[index],(index,ds,ori_len))
        bundle.append((index,ds,ori_len))
    return R1_itnpl,R1,R1_bd_itnpl,np.repeat(R1_bd,n_ts,axis=0),bundle
def gen_ds_R1_ext(R1,sample_num,R1_bd,sample_points=None,sample_pattern=None,augmentation=False,rise_index=None,OUT_DIM=None,average=False,debug=False,knownbundle=None):
    """
    input shape: (batch,OUT_DIM)
    output shape: (batch,sample_num)
    """    
    R1 = np.concatenate((np.repeat(R1[:,0:1],40,axis=1),R1,np.repeat(np.reshape(R1[:,-1],(-1,1)),40,axis=1)),axis=-1)
    R1_bd = np.concatenate((np.repeat(R1_bd[:,0:1],40,axis=1),R1_bd,np.repeat(np.reshape(R1_bd[:,-1],(-1,1)),40,axis=1)),axis=-1)
    if augmentation == False:
        s_time = 60//2
    else:
#         s_time = np.random.randint(40//2,62//2,1)[0]
#         s_time = np.random.randint(20//2, 40//2, 1)[0]
        s_time = np.random.randint(55//2,65//2,1)[0]
#         print(s_time)
#         s_time = 60//2
    if sample_pattern == "RES01":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(80)//2.9
        delayed_venous_index = post_venous_index +(80)//2.9
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index],dtype=np.uint16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)     
    elif sample_pattern == "Random":

        arterial_index = np.argmax(R1_bd)

        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        sample_num = np.random.randint(4,7)
        ds = ori_len//sample_num
        ds_post = (OUT_DIM-s_time)//(sample_num-1)
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        index = []
        index.append(pre_index)
        for inds in range(sample_num-1):
            index.append(s_time+inds*ds_post)
            
        index = np.array(index)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            _to_return = R1[:,index]
            return _to_return,R1,(index,ds,ori_len),R1_bd[:,index],R1_bd
#     elif sample_pattern == "Bundle":
#         arterial_index = np.argmax(R1_bd)
#         R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
#         R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation

#         index,ds,ori_len  = knownbundle
#         sample_num = len(index)
#         ds_post = (ori_len-30-arterial_index)//sample_num 
#         if rise_index == None:
#             rise_index = search_rise_tp(R1_bd)
#         if average == True:
#             weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
#             R1_ave = np.zeros((R1.shape[0],5))
#             R1_ave[:,0] = R1[:,index[0]]
#             for ind in range(1,4):
#                 to_return = R1[:,index]
# #                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
#                 R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
#             return R1_ave,R1,(index,ds,ori_len)
#         else:
#             return R1[:,index],R1,(index,ds,ori_len) 
        
def gen_ds_R1_time_shift(R1,sample_num,R1_bd,sample_points=None,sample_pattern=None,augmentation=False,rise_index=None,OUT_DIM=None,average=False,debug=False,time_shift=None):
    s_time = 60//2
    arterial_index = np.argmax(R1_bd)+time_shift//2
    R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
    R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
    ori_len = R1.shape[-1]
    ds = ori_len//sample_num
    if rise_index == None:
        rise_index = search_rise_tp(R1_bd)
    pre_index = 0
    R1[:,pre_index] = 0
    arterial_index = np.argmax(R1_bd)+time_shift//2
    post_venous_index = arterial_index +(30)//2 
    delayed_venous_index = post_venous_index +(30)//2
    index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index])
    return R1[:,index],R1,(index,ds,ori_len)
        
def gen_ds_R1(R1,sample_num,R1_bd,sample_points=None,sample_pattern=None,augmentation=False,rise_index=None,OUT_DIM=None,average=False,debug=False):
    """
    input shape: (batch,OUT_DIM)
    output shape: (batch,sample_num)
    """    
    if augmentation == False:
        s_time = 60//2
    else:
#         s_time = np.random.randint(40//2,62//2,1)[0]
#         s_time = np.random.randint(20//2, 40//2, 1)[0]
        s_time = np.random.randint(55//2,65//2,1)[0]
#         print(s_time)
#         s_time = 60//2
    if debug == True:
        s_time = 60//2
    if sample_pattern == "pancreas":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        arterial_index2 = arterial_index+10//2
        post_venous_index = arterial_index +(10+10+20)//2 
        delayed_venous_index = post_venous_index +(10+20)//2
        index = np.array([pre_index,arterial_index,arterial_index2,post_venous_index,delayed_venous_index])
        if average == True:
            weighting = np.array([0.2,0.4,0.6,0.8,1,0.8,0.6,0.4,0.2])/5.0
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,5):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
        
    if sample_pattern == "pancreas_new":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(10+20)//2 
        delayed_venous_index = post_venous_index +(10+20)//2
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index])
        if average == True:
            weighting = np.array([0.2,0.4,0.6,0.8,1,0.8,0.6,0.4,0.2])/5.0
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "RES04":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        arterial2_index = arterial_index + 8//2
        post_venous_index = arterial2_index +(34)//2 
        delayed_venous_index = post_venous_index +(36)//2
        index = np.array([pre_index,arterial_index,arterial2_index,post_venous_index,delayed_venous_index])
        if average == True:
            weighting = np.array([0.2,0.4,0.6,0.8,1,0.8,0.6,0.4,0.2])/5.0
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "RES01":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(30)//2 
        delayed_venous_index = post_venous_index +(30)//2
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index])
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "RES02":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        arterial2_index = arterial_index + (7.6//2)
        post_venous_index = arterial2_index +(32.3//2)
        delayed_venous_index = post_venous_index +(32.2//2)
        index = np.array([pre_index,arterial_index,arterial2_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "DCE02":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        arterial2_index = arterial_index + (9.3//2)
        post_venous_index = arterial2_index +(34//2)
        delayed_venous_index = post_venous_index +(34//2)
        index = np.array([pre_index,arterial_index,arterial2_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "DCE05":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        arterial2_index = arterial_index + (8//2)
        post_venous_index = arterial2_index +(36//2)
        delayed_venous_index = post_venous_index +(36//2)
        index = np.array([pre_index,arterial_index,arterial2_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "RES11":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(40//2 )
        delayed_venous_index = post_venous_index +(43.1//2)
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "DCE01":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(32//2 )
        delayed_venous_index = post_venous_index +(32//2)
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "DCE03":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(33//2 )
        delayed_venous_index = post_venous_index +(33.2//2)
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    if sample_pattern == "DCE04":
        arterial_index = np.argmax(R1_bd)
        R1 = R1[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        R1_bd = R1_bd[:,(arterial_index-s_time):(arterial_index-s_time+OUT_DIM)] #extrapolation
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
#         R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        R1[:,pre_index] = 0
        arterial_index = np.argmax(R1_bd)
        post_venous_index = arterial_index +(36//2 )
        delayed_venous_index = post_venous_index +(36//2)
        index = np.array([pre_index,arterial_index,post_venous_index,delayed_venous_index]).astype(np.int16)
        if average == True:
            weighting = np.array([0.1,0.3,0.5,0.8,1,0.8,0.5,0.3,0.1])/4.4
            R1_ave = np.zeros((R1.shape[0],5))
            R1_ave[:,0] = R1[:,index[0]]
            for ind in range(1,4):
#                 print(R1[:,index[ind]-10//2+1:(index[ind]+10//2)].shape)
                R1_ave[:,ind] = R1[:,index[ind]-10//2+1:(index[ind]+10//2)].dot(weighting)
            return R1_ave,R1,(index,ds,ori_len)
        else:
            return R1[:,index],R1,(index,ds,ori_len)
    elif sample_pattern == None:
        ori_len = R1.shape[-1]
        ds = ori_len//sample_num
        index = range(0,ori_len,ds)
        return R1[:,index],(index,ds,ori_len)
        
#     if sample_points == None:
#         ds = ori_len//sample_num
#         index = range(0,ori_len,ds)
#         return R1[:,index],(index,ds,ori_len)
#     else:
#         ds = ori_len//sample_num
#         return R1[:,sample_points],(sample_points,ds,ori_len)
def gen_s_2_ca(St,R10,r):
    return (St-St[:,0])/(St[:,0]*r*R10)

def gen_ds_R1_wt_timing(R1,sample_num,R1_bd,sample_points=None,sample_pattern=None,augmentation=None,rise_index=None):
    """
    input shape: (batch,OUT_DIM)
    output shape: (batch,sample_num)
    """
    ori_len = R1.shape[-1]
    if sample_pattern == "pancreas":
        ds = ori_len//sample_num
        if rise_index == None:
            rise_index = search_rise_tp(R1_bd)
#         pre_index = rise_index
        pre_index = 0
        R1[:,pre_index] = np.mean(R1[:,0:rise_index],axis=-1)
        arterial_index = np.argmax(R1_bd)
        arterial_index2 = arterial_index+9
        post_venous_index = arterial_index +9 +(9+10)//2 +20
        delayed_venous_index = post_venous_index +(10+10)//2 +20
        index = np.array([pre_index,arterial_index,arterial_index2,post_venous_index,delayed_venous_index])
        seq = np.zeros([ori_len])
        seq[index[0]:index[1]]=1
        seq[index[1]:index[2]]=2
        seq[index[2]:index[3]]=3
        seq[index[3]:index[4]]=4
        seq[index[4]:ori_len]=5
        return R1[:,index],(index,ds,ori_len),seq
    elif sample_pattern == None:
        ds = ori_len//sample_num
        index = range(0,ori_len,ds)
        return R1[:,index],(index,ds,ori_len)
def gen_inge_spike(R1_ds,bundle,OUT_DIM=None):
    R1_ds=np.reshape(R1_ds,(-1,R1_ds.shape[-1]))
    (index,ds,ori_len) = bundle
    if OUT_DIM == None:
        OUT_DIM = ori_len
    spiked = np.zeros((R1_ds.shape[0],ori_len))
    inlen = len(index)
    for ind in range(1,inlen):
        spiked[:,index[ind]] = R1_ds[:,ind]

    spiked[:,0] = R1_ds[:,0]
    
    return spiked
def gen_intpl(R1_ds,bundle):
    (index,ds,ori_len) = bundle
    from scipy import interpolate
    f_interp = interpolate.interp1d(index,R1_ds,bounds_error=False,fill_value=(R1_ds[:,0],R1_ds[:,-1]))
    return f_interp(range((R1_ds.shape[-1])*ds))
def gen_del_R1(R1,R1_bd):
    """
    input/output shape: (batch, DCE_DIM)
    """
    rise_index = search_rise_tp(R1_bd)
    R_0 = np.mean(R1[:,0:rise_index],axis=-1)
    del_R1 = (R1 - R_0.reshape(R_0.shape[-1],-1))
    return del_R1
def gen_del_S(S1,S1_bd):
    """
    input/output shape: (batch, DCE_DIM)
    """
    rise_index = search_rise_tp(S1_bd)
    R_0 = np.mean(S1[:,0:rise_index],axis=-1)
    del_R1 = (S1 - R_0.reshape(R_0.shape[-1],-1))/R_0.reshape(R_0.shape[-1],-1)
    del_R1[del_R1<1e-4]=1e-4
    return del_R1
def process_nn (h5file):
    kv = np.array(h5file['kv'],dtype=np.float32).transpose(1,0)
    R10 = np.array(h5file['T1'],dtype=np.float32).transpose(1,0)[:,0]
    bd_R10 = np.mean(np.array(h5file['bd_T1'],dtype=np.float32).transpose(1,0)[:,0])
    S_ts = np.array(h5file['S_im'],dtype=np.float32).transpose(1,0)
    S_ts = (S_ts-S_ts[:,0].reshape(-1,1))/S_ts[:,0].reshape(-1,1)
    S_bd = np.array(h5file['S_bd'],dtype=np.float32).transpose(1,0)
    S_bd = (S_bd-S_bd[:,0].reshape(-1,1))/S_bd[:,0].reshape(-1,1)
    C_bd_lbl =  np.array(h5file['c_pl'],dtype=np.float32).transpose(1,0)


    # generate downsampled S
    S_ts_ds,S_ts,bundle = gen_ds_R1(S_ts,INIT_DIM,C_bd_lbl,OUT_DIM=OUT_DIM,sample_pattern='pancreas_new',average=False)
    # generate blood input 
    S_bd_ds,S_bd,bundle = gen_ds_R1(S_bd,INIT_DIM,C_bd_lbl,OUT_DIM=OUT_DIM,sample_pattern='pancreas_new',average=False)
    
    # generate spike
    S_ts_itnpl = gen_inge_spike(S_ts_ds,bundle)
    S_bd_itnpl = gen_inge_spike(S_bd_ds,bundle)

    bd_temp = np.repeat(S_bd_itnpl.reshape([-1,OUT_DIM]),S_ts.shape[0],axis=0)
    bd_temp2 = np.repeat(S_bd.reshape([-1,OUT_DIM]),S_ts.shape[0],axis=0)
    
    X_test_ts = S_ts_itnpl
    Y_test_ts = S_ts
    
    X_test_bd = bd_temp
    Y_test_bd = bd_temp2
    return X_test_ts,X_test_bd,Y_test_ts,Y_test_bd

def prepare_nn_ds(data,seg,ori_len=180,s_time = 15,t_res=1.4):
    """
    process image into voxels 
    input: 
        data - data dictionary {'data','series_name','acq_time'}
        seg - segmentation with label 1 as tumor, 2 as nontumor, 3 as left ventricular, 4 as lymphs (if any), 5 as other tumors (if any) 
    
    output: 
        inp_tum - concatenated tumor signal and AIF (baseline-corrected, normalized by max of AIF)
        inp_nontum
    """

#     n_img = len(data)
#     inp_ori = [data[i]['data'] for i in range(n_img)]
#     inp_ori = np.transpose(np.asarray(inp_ori),(1,2,3,0))
#     inp_tum_ori,inp_nontum_ori,inp_aif_ori = inp_ori[seg==1],inp_ori[seg==2],inp_ori[seg==3] 
#     inp_aif = np.mean(inp_aif_ori,axis=0,keepdims=True)
    
#     inp_tum = np.concatenate((inp_tum_ori,inp_aif),axis=0)
#     inp_nontum = np.concatenate((inp_nontum_ori,inp_aif),axis=0)
    
#     norm_constant = np.max(inp_aif)
#     print(norm_constant)
#     inp_tum = (inp_tum - inp_tum[:,0].reshape(-1,1))/norm_constant
#     inp_nontum = (inp_nontum - inp_nontum[:,0].reshape(-1,1))/norm_constant

    
#     pre_index = 0
#     index=[]
#     index.append(int(pre_index))
#     index.append(int(s_time))
#     for inds in range(2,n_img): # for phase 1 to the end 
#         ds_post = time_diff_json(data[inds-1]['acq_time'],data[inds]['acq_time'])
#         #correct for potential error 
#         if ds_post>200 and inds!=n_img-1:
#             print(inds,n_img,ds_post)
#             ds_post = time_diff_json(data[inds]['acq_time'],data[inds+1]['acq_time'])
#         index.append(int(s_time+(ds_post//t_res))) #need ds_post from header 
#         s_time = s_time+(ds_post//t_res)
        
#     inp_tum = gen_inge_spike(inp_tum,(index,ori_len//n_img,ori_len))
#     inp_nontum = gen_inge_spike(inp_nontum,(index,ori_len//n_img,ori_len))

    """
    version 2.0
    use the last gap as a constant gap to match training data
    interpolate "non-exist" phases 
    """
#     from scipy import interpolate
    
#     n_img = len(data)
#     inp_ori = [data[i]['data'] for i in range(n_img)]
#     inp_ori = np.transpose(np.asarray(inp_ori),(1,2,3,0))
#     inp_tum_ori,inp_nontum_ori,inp_aif_ori = inp_ori[seg==1],inp_ori[seg==2],inp_ori[seg==3] 
#     inp_aif = np.mean(inp_aif_ori,axis=0,keepdims=True)
    
#     inp_tum = np.concatenate((inp_tum_ori,inp_aif),axis=0)
#     inp_nontum = np.concatenate((inp_nontum_ori,inp_aif),axis=0)
    
#     norm_constant = np.max(inp_aif)
#     inp_tum = (inp_tum - inp_tum[:,0].reshape(-1,1))/norm_constant
#     inp_nontum = (inp_nontum - inp_nontum[:,0].reshape(-1,1))/norm_constant

#     ds_post = time_diff_json(data[-2]['acq_time'],data[-1]['acq_time'])
#     pre_index = 0
#     index=[]
#     index.append(int(pre_index))
#     index.append(int(s_time))
#     for inds in range(2,n_img): # for phase 2 to the end 
#         index.append(int(s_time+(ds_post//t_res))) #need ds_post from header 
#         s_time = s_time+(ds_post//t_res)
    
#     new_n_img = int((ori_len-index[1]-1)//(ds_post/t_res))+2
#     if new_n_img > n_img:
#         new_index = index.copy()
        
#         for inds in range(n_img,new_n_img): # for phase 2 to the end 
#             new_index.append(int(s_time+(ds_post//t_res))) #need ds_post from header 
#             s_time = s_time+(ds_post//t_res)
#         new_inp_tum = np.zeros((inp_tum.shape[0],new_n_img))
#         new_inp_tum[:,0:n_img] = inp_tum
#         for _n_inp_tum in range(inp_tum.shape[0]):
#             f = interpolate.interp1d(index, inp_tum[_n_inp_tum,:], fill_value='extrapolate')    
#             new_inp_tum[_n_inp_tum,n_img:new_n_img] = f(new_index[n_img:new_n_img])
#         new_inp_nontum = np.zeros((inp_nontum.shape[0],new_n_img))
#         new_inp_nontum[:,0:n_img] = inp_nontum
#         for _n_inp_nontum in range(inp_nontum.shape[0]):
#             f = interpolate.interp1d(index, inp_nontum[_n_inp_nontum,:], fill_value='extrapolate')    
#             new_inp_nontum[_n_inp_nontum,n_img:new_n_img] = f(new_index[n_img:new_n_img])
#         inp_tum = gen_inge_spike(new_inp_tum,(new_index,ds_post//t_res,ori_len))
#         inp_nontum = gen_inge_spike(new_inp_nontum,(new_index,ds_post//t_res,ori_len))

#     elif new_n_img == n_img:
#         inp_tum = gen_inge_spike(inp_tum,(index,(ds_post//t_res),ori_len))
#         inp_nontum = gen_inge_spike(inp_nontum,(index,(ds_post//t_res),ori_len))
#     elif new_n_img < n_img: 
#         inp_tum = gen_inge_spike(inp_tum[:,0:new_n_img],(index[0:new_n_img],(ds_post//t_res),ori_len))
#         inp_nontum = gen_inge_spike(inp_nontum[:,0:new_n_img],(index[0:new_n_img],(ds_post//t_res),ori_len))
#     return inp_tum,inp_nontum
    """
    version 3.0
    use the real gap 
    use c(t)=(s(t)-s(0))/s(0) instead of sbd(0)
    interpolate "non-exist" phases 
    """
    from scipy import interpolate
    
    n_img = len(data)
    inp_ori = [data[i]['data'] for i in range(n_img)]
    inp_ori = np.transpose(np.asarray(inp_ori),(1,2,3,0))
    inp_tum_ori,inp_nontum_ori,inp_aif_ori = inp_ori[seg==1],inp_ori[seg==2],inp_ori[seg==3] 
    inp_aif = np.mean(inp_aif_ori,axis=0,keepdims=True)
    
    inp_tum = np.concatenate((inp_tum_ori,inp_aif),axis=0)
    inp_nontum = np.concatenate((inp_nontum_ori,inp_aif),axis=0)
    
    norm_constant = np.max(inp_aif)
    inp_tum = (inp_tum - inp_tum[:,0].reshape(-1,1))/(inp_tum[:,0].reshape(-1,1))/norm_constant
    inp_nontum = (inp_nontum - inp_nontum[:,0].reshape(-1,1))/(inp_nontum[:,0].reshape(-1,1))/norm_constant

    
    pre_index = 0
    index=[]
    index.append(int(pre_index))
    index.append(int(s_time))
    for inds in range(2,n_img): # for phase 2 to the end 
        ds_post = time_diff_json(data[inds-1]['acq_time'],data[inds]['acq_time'])
        index.append(int(s_time+(ds_post//t_res))) #need ds_post from header 
        s_time = s_time+(ds_post//t_res)
    
    new_n_img = int((ori_len-index[1]-1)//(ds_post/t_res))+2
    if new_n_img > n_img:
        new_index = index.copy()
        
        for inds in range(n_img,new_n_img): # for phase 2 to the end 
            new_index.append(int(s_time+(ds_post//t_res))) #need ds_post from header 
            s_time = s_time+(ds_post//t_res)
        new_inp_tum = np.zeros((inp_tum.shape[0],new_n_img))
        new_inp_tum[:,0:n_img] = inp_tum
        for _n_inp_tum in range(inp_tum.shape[0]):
            f = interpolate.interp1d(index, inp_tum[_n_inp_tum,:], fill_value='extrapolate')    
            new_inp_tum[_n_inp_tum,n_img:new_n_img] = f(new_index[n_img:new_n_img])
        new_inp_nontum = np.zeros((inp_nontum.shape[0],new_n_img))
        new_inp_nontum[:,0:n_img] = inp_nontum
        for _n_inp_nontum in range(inp_nontum.shape[0]):
            f = interpolate.interp1d(index, inp_nontum[_n_inp_nontum,:], fill_value='extrapolate')    
            new_inp_nontum[_n_inp_nontum,n_img:new_n_img] = f(new_index[n_img:new_n_img])
        inp_tum = gen_inge_spike(new_inp_tum,(new_index,ds_post//t_res,ori_len))
        inp_nontum = gen_inge_spike(new_inp_nontum,(new_index,ds_post//t_res,ori_len))

    elif new_n_img == n_img:
        inp_tum = gen_inge_spike(inp_tum,(index,(ds_post//t_res),ori_len))
        inp_nontum = gen_inge_spike(inp_nontum,(index,(ds_post//t_res),ori_len))
    elif new_n_img < n_img: 
        inp_tum = gen_inge_spike(inp_tum[:,0:new_n_img],(index[0:new_n_img],(ds_post//t_res),ori_len))
        inp_nontum = gen_inge_spike(inp_nontum[:,0:new_n_img],(index[0:new_n_img],(ds_post//t_res),ori_len))
    return inp_tum,inp_nontum