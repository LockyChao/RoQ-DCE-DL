"""
network training functions
"""
import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../python/")
from lib.simu import *
from scipy.signal import convolve
# def DCloss_mag(y_pred,y,x):
#     mask = tf.math.greater(x,0)
#     return tf.math.reduce_mean(tf.math.square(tf.math.multiply(y - y_pred,mask)))
def DCloss_mag(y_pred,x,reg=0.01):
    mask = tf.math.greater(x,0)
    N_ind = tf.reduce_sum(tf.cast(mask, tf.float32))
    y_pred_masked = tf.keras.layers.Reshape((-1,1))(tf.boolean_mask(y_pred, mask))
    y_masked = tf.keras.layers.Reshape((-1,1))(tf.boolean_mask(x, mask))
    mse = tf.math.reduce_sum(tf.math.square(y_masked-y_pred_masked))/(N_ind+0.01)
    
    return reg*mse

def Smoothloss(y_pred,reg=0.01):
    y_expand = tf.expand_dims(y_pred,axis=-1)
#     print(y_expand.shape)
    loss = tf.reduce_sum(tf.image.total_variation(y_expand))
    
    return reg*loss
# def len_tofts_tf (ktr,ve,n,del_t):
#     return tf.math.exp(tf.reshape((ktr/ve),(-1,1)),(tf.reshape(n,(1,-1))*del_t))
# def con_tofts_tf(c,vp,ve,ktr):
#     """
#     input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
#     output: c_new (N_pixel,N_temporal)
#     """
#     del_t = 2.0
#     n = np.array(range(c.shape[-1])).astype(np.float32)
#     c_new = tf.reshape(vp,(-1,1))*c+
#              (tf.math.exp(-tf.matmul(tf.reshape(ktr/ve,(-1,1)),tf.reshape(n,(1,-1)))*del_t))
#              *tf.math.cumsum(c*n*len_tofts_tf(ktr,ve,n,del_t)*del_t,axis=-1)
#     return c_new

# def my_conv(x1,x2):
#     return np.convolve(x1,x2,mode='full')
# @tf.function(input_signature=[tf.TensorSpec(None, tf.float32),tf.TensorSpec(None, tf.float32)])
# def tf_conv(x1,x2):
#     y = tf.numpy_function(my_conv, [x1,x2], [tf.float32,tf.float32])
#     return y


def len_tofts_tf (ktr,ve,n,del_t):
    return tf.math.exp(tf.matmul(tf.cast(tf.reshape((ktr/ve),(-1,1)),tf.float32),tf.cast(tf.reshape(n,(1,-1)),tf.float32))*del_t)
def con_oritofts_tf_fit(c,ktr,ve,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    ktr = ktr
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-ktr,ve,n,del_t)
    c_new=ktr*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t

    return tf.reshape(c_new,tf.shape(c)).numpy().flatten()
def con_oritofts_kep_tf(c,ktr,kep,del_t):
    """
    
    for dl00 tf1.15, some changes 
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    ktr = ktr
    n = range(c.shape[-1])
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-kep,1,n,del_t)
    c_new=ktr*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t

    return tf.reshape(c_new,tf.shape(c))
def con_tofts_kep_tf_fit(c,vp,kep,ktr,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-kep,1,n,del_t)
    c_new=vp*c+ktr*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t

    return tf.reshape(c_new,tf.shape(c)).numpy().flatten()
def con_oritofts_kep_tf_fit(c,ktr,kep,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    ktr = ktr
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-kep,1,n,del_t)
    c_new=ktr*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t

    return tf.reshape(c_new,tf.shape(c)).numpy().flatten()
def f_VARPRO(c,kep,del_t):
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-kep,1,n,del_t)
    c_new=tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t
#     print('f ',tf.reshape(c_new,tf.shape(c)).numpy().flatten())
    return tf.reshape(c_new,tf.shape(c)).numpy().flatten()
def g_VARPRO(c,c_t,kep,del_t):
    """
    parameterize ktrans to a function of kep 
    """
    
    upside = tf.cast(tf.math.reduce_sum(np.array(c_t).flatten()*f_VARPRO(c,kep,del_t)),tf.float32)
#     print('up ',upside)
    downside = tf.cast(tf.math.reduce_sum(f_VARPRO(c,kep,del_t)**2),tf.float32)
#     print('down ',downside)
#     print('g ',tf.cast(upside/downside,tf.float32))
    return tf.cast(upside/downside,tf.float32)
    
def fun_oritofts_VARPRO(c_p,c_t,kep,del_t,reg=1):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    kep = tf.cast(kep,tf.float32)
    c_p = tf.cast(c_p,tf.float32)
    c_t = tf.cast(c_t,tf.float32)
    lhs = c_t 
    rhs = g_VARPRO(c_p,c_t,kep,del_t)*f_VARPRO(c_p,kep,del_t)
#     print('rhs ',tf.shape(g_VARPRO(c_p,c_t,kep)*f_VARPRO(c_p,kep)))
    return reg*(lhs - rhs).numpy().flatten()
def f_VARPRO_np(c,kep,del_t):
    n = range(c.shape[-1])
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts (-kep,1,n,del_t)
    c_new=np.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t
#     print('f ',tf.reshape(c_new,tf.shape(c)).numpy().flatten())
    return tf.reshape(c_new,tf.shape(c)).numpy().flatten()
def g_VARPRO_np(c,c_t,kep):
    """
    parameterize ktrans to a function of kep 
    """
    
    upside =np.sum(np.array(c_t).flatten()*f_VARPRO_np(c,kep)).astype(np.float32)
#     print('up ',upside)
    downside = np.sum(f_VARPRO_np(c,kep)**2).astype(np.float32)
#     print('down ',downside)
#     print('g ',tf.cast(upside/downside,tf.float32))
    return (upside/downside).astype(np.float32)
    
def fun_oritofts_VARPRO_np(c_p,c_t,kep,del_t,reg=1):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    kep = kep.astype(np.float32)
    c_p = c_p.astype(np.float32)
    c_t = c_t.astype(np.float32)
    lhs = c_t 
    rhs = g_VARPRO_np(c_p,c_t,kep)*f_VARPRO_np(c_p,kep)
#     print('rhs ',tf.shape(g_VARPRO(c_p,c_t,kep)*f_VARPRO(c_p,kep)))
    return reg*(lhs - rhs).flatten()
def con_tofts_tf(c,vp,ve,ktr,pad_num,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    lenc = pad_num+1
    c = tf.reshape(c,(1,lenc))
    ktr = tf.cast(ktr,tf.float32)
    n = tf.constant(range(c.shape[-1]))
    p1 = tf.cast(tf.reshape(vp,(-1,1))*c,tf.float32)
    """
    v1
    """
#     p2 = tf.reshape(ktr,(-1,1))*tf.math.exp(tf.matmul(tf.reshape((-ktr/ve),(-1,1)),tf.cast(tf.reshape(n,(1,-1)),tf.float32))*del_t)
#     p3 = tf.math.cumsum(c*len_tofts_tf(ktr,ve,n,del_t),axis=-1)*del_t
#     c_new = p1+p2*p3
    """
    v2
    """
#     inp = tf.concat([c,len_tofts_tf(-ktr,ve,n,del_t)],axis=-1)
#     length = c.shape[-1]
# #     c_new = p1+tf.reshape(ktr,(-1,1))*tf.keras.layers.Lambda(lambda m: np.convolve(m[:length],m[length:],mode='full')[:length])(inp)*del_t
#     c_new = p1+tf.reshape(ktr,(-1,1))*tf.map_fn(lambda m: np.convolve(m[:length],m[length:],mode='full')[:length],inp)*del_t

    """
    v3
    """
#     inp = tf.concat([c,len_tofts_tf(-ktr,ve,n,del_t)],axis=-1)
#     length = c.shape[-1]
# #     tf.map_fn(lambda t:print(tf.expand_dims(tf.expand_dims(t[:length],axis=-1),axis=0).shape),inp)
#     tf.map_fn(lambda m:print(tf.expand_dims(tf.expand_dims(m[length:],axis=-1),axis=-1).shape),inp)
#     c_new = p1+tf.reshape(ktr,(-1,1))*tf.map_fn(fn=func_lam,elems=inp)*del_t
    """
    v4, batch size 1
    """
    length = c.shape[-1]
    padding = [[0,0],[pad_num,0],[0,0]]
    b = tf.cast(len_tofts_tf (-ktr,ve,n,del_t),tf.float32)
    c_new=p1+ktr*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t

    return tf.reshape(c_new,tf.shape(c))

def Toftsloss(y_pred,c_ts,c_pl,reg,pad_num=69):
    """
    to match target kv order (vp,ktr,ve) 
    """
    pd = con_tofts_tf(c_pl,y_pred[:,0],y_pred[:,1],y_pred[:,2],pad_num=pad_num)
    return reg*tf.math.reduce_mean(tf.math.squared_difference(pd, c_ts))

def con_patlak_tf(c,vp,ktr,pad_num,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    ktr = tf.cast(ktr,tf.float32)
    vp = tf.cast(vp,tf.float32)
    c = tf.cast(c,tf.float32)
    ktr = ktr
#     n = tf.constant(range(c.shape[-1]))
    p1 = tf.reshape(vp,(-1,1))*c
    length = c.shape[-1]
    padding = [[0,0],[pad_num,0],[0,0]]
    c_new=p1+ktr*tf.reshape(tf.math.cumsum(c,axis=-1),(1,-1))*del_t
    return c_new
def Patlakloss(y_pred,c_ts,c_pl,reg,pad_num=69):
    """
    to match target kv order (vp,ktr) 
    """
    pd = con_patlak_tf(c_pl,y_pred[:,0],y_pred[:,1],pad_num)
    return reg*tf.math.reduce_mean(tf.math.squared_difference(pd, c_ts))

def con_oritofts_tf(c,ktr,ve,pad_num,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    ktr = ktr
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-ktr,ve,n,del_t)
    c_new=ktr*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t

    return tf.reshape(c_new,tf.shape(c))

def f_VARPRO_tf(c,kep,del_t):
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    pad_num = length-1
    padding = [[0,0],[pad_num,0],[0,0]]
    b = len_tofts_tf (-kep,1,n,del_t)
    c_new=tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t
#     print('f ',tf.reshape(c_new,tf.shape(c)).numpy().flatten())
    return tf.keras.layers.Flatten()(c_new)

def g_VARPRO_tf(c,c_t,kep,del_t):
    """
    parameterize ktrans to a function of kep 
    """
    
    upside = tf.cast(tf.math.reduce_sum(tf.keras.layers.Flatten()(c_t)*f_VARPRO_tf(c,kep,del_t)),tf.float32)
#     print('up ',upside)
    downside = tf.cast(tf.math.reduce_sum(f_VARPRO_tf(c,kep,del_t)**2),tf.float32)
#     print('down ',downside)
#     print('g ',tf.cast(upside/downside,tf.float32))
    return tf.cast(upside/(downside+1e-12),tf.float32)
    
def f_VARPROloss(c_p,c_t,kep,del_t):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    kep = tf.cast(kep,tf.float32)
    c_p = tf.cast(c_p,tf.float32)
    c_t = tf.cast(c_t,tf.float32)
    lhs = c_t 
    rhs = g_VARPRO_tf(c_p,c_t,kep,del_t)*f_VARPRO_tf(c_p,kep,del_t)
#     print('rhs ',tf.shape(g_VARPRO(c_p,c_t,kep)*f_VARPRO(c_p,kep)))
    return tf.keras.layers.Flatten()((lhs - rhs))

def f_VARPROloss_DC(c_p,c_t,kep,c_ts_inp,del_t):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    kep = tf.cast(kep,tf.float32)
    c_p = tf.cast(c_p,tf.float32)
    c_t = tf.cast(c_t,tf.float32)
    lhs = c_t 
    rhs = g_VARPRO_tf(c_p,c_t,kep,del_t)*f_VARPRO_tf(c_p,kep,del_t)
#     print('rhs ',tf.shape(g_VARPRO(c_p,c_t,kep)*f_VARPRO(c_p,kep)))
    return tf.keras.layers.Flatten()((lhs - rhs))+DCloss_mag(rhs, c_ts_inp,reg=1)

def f_VARPROloss_DC_debug_onlythird(c_p,c_t,kep,c_ts_inp,del_t):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    kep = tf.cast(kep,tf.float32)
    c_p = tf.cast(c_p,tf.float32)
    c_t = tf.cast(c_t,tf.float32)
    lhs = c_t 
    rhs = g_VARPRO_tf(c_p,c_t,kep)*f_VARPRO_tf(c_p,kep)
#     print('rhs ',tf.shape(g_VARPRO(c_p,c_t,kep)*f_VARPRO(c_p,kep)))
    return 0*tf.keras.layers.Flatten()((lhs - rhs))+DCloss_mag(rhs, c_ts_inp,reg=1)

def f_VARPROloss_DC_debug_onlysecond(c_p,c_t,kep,c_ts_inp,del_t):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    kep = tf.cast(kep,tf.float32)
    c_p = tf.cast(c_p,tf.float32)
    c_t = tf.cast(c_t,tf.float32)
    lhs = c_t 
    rhs = g_VARPRO_tf(c_p,c_t,kep)*f_VARPRO_tf(c_p,kep)
#     print('rhs ',tf.shape(g_VARPRO(c_p,c_t,kep)*f_VARPRO(c_p,kep)))
    return tf.keras.layers.Flatten()((lhs - rhs))+0*DCloss_mag(rhs, c_ts_inp,reg=1)

def o_Toftsloss(y_pred,c_ts,c_pl,reg,pad_num,mode='ve',c_ts_inp=None,del_t=None):
    """
    to match target kv order (ktr,ve) 
    """
    if mode == 've':
        pd = con_oritofts_tf(c_pl,y_pred[:,0],y_pred[:,1],pad_num,del_t=del_t)
    elif mode == 'kep':
        pd = con_oritofts_kep_tf(c_pl,y_pred[:,0],y_pred[:,1],del_t=del_t)
    elif mode == 'VARPRO':
        return reg*tf.math.reduce_mean(f_VARPROloss(c_pl,c_ts,y_pred[:,0],del_t=del_t)**2)
    elif mode == 'VARPRODC':
        return reg*tf.math.reduce_mean(f_VARPROloss_DC(c_pl,c_ts,y_pred[:,0],c_ts_inp,del_t=del_t)**2)
    elif mode == 'VARPRODC_onlythird':
        return reg*tf.math.reduce_mean(f_VARPROloss_DC_debug_onlythird(c_pl,c_ts,y_pred[:,0],c_ts_inp,del_t=del_t)**2)
    elif mode == 'VARPRODC_onlysecond':
        return reg*tf.math.reduce_mean(f_VARPROloss_DC_debug_onlysecond(c_pl,c_ts,y_pred[:,0],c_ts_inp,del_t=del_t)**2)
    elif mode == 'fit':
        from tfg.math.optimizer.levenberg_marquardt import minimize 
        def fun_to_fit_VARPRO(x):
            return tf.reshape(f_VARPROloss(c_pl, c_ts,x,del_t=del_t),(-1))
        result = minimize(residual = fun_to_fit_VARPRO, variables = np.array([0.01]),max_iterations=30)
        return result
    return reg*tf.math.reduce_mean(tf.math.squared_difference(pd, c_ts))

def con_2cxm_tf(c,Fp,A,alpha,beta,del_t):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    n = tf.constant(range(c.shape[-1]))
    length = c.shape[-1]
    padding = [[0,0],[69,0],[0,0]]
    b1 = A*len_tofts_tf (-alpha,1,n,del_t)
    b2 = (1.0-A)*len_tofts_tf(-beta,1,n,del_t)
    b=b1+b2
    c_new=Fp*tf.reshape(tf.nn.conv1d(tf.cast(tf.pad(tf.reshape(c,(1,-1,1)),padding),tf.float32),tf.reverse(tf.cast(tf.reshape(b,(-1,1,1)),tf.float32),axis=[0]),stride=1,padding='VALID'),(1,-1))*del_t
    
    return c_new
    
def TwoCompartExloss(y_pred,c_ts,c_pl,reg):
    """
    (Fp,A,alpha,beta)
    """
    pd = con_2cxm_tf(c_pl,y_pred[:,0],y_pred[:,1],y_pred[:,2],y_pred[:,3])
    return reg*tf.math.reduce_mean(tf.math.squared_difference(pd, c_ts))

def UpsampModel():
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    IN_DIM = 70
    tis_inp = tf.keras.Input(shape=(IN_DIM), name="tissue_ds")
    art_inp = tf.keras.Input(shape=(IN_DIM), name="aorta_fs")

    with tf.name_scope("concat1"):
        x1 = tf.keras.layers.Concatenate(axis=-1)([tis_inp, art_inp])
    with tf.name_scope("upsample"):
        x2 = tf.keras.layers.Dense(units=IN_DIM)(x1)
        x2 = tf.keras.layers.Dense(units=IN_DIM,activation='relu')(x2)
        f_ts = tf.keras.layers.Dense(units=IN_DIM,name="tissue_curve",activation='relu')(x2)
    
        x3 = tf.keras.layers.Dense(units=IN_DIM)(x1)
        x3 = tf.keras.layers.Dense(units=IN_DIM,activation='relu')(x3)
        f_bd = tf.keras.layers.Dense(units=IN_DIM,name="aorta_curve",activation='relu')(x3)
    model = Model(inputs=[tis_inp,art_inp],outputs=[f_ts,f_bd])
    return model
    
def apply_transformations(image, label,SCALE=True,NOISE=True,SHIFT=True):
    """
    Applys augmentations to a (image, label) pair.
    
    Args:
     image : a tensor image
     label : a tensor [label for the image]

    """
    inp_ts = image['tissue_ds']
    inp_bd = image['aorta_fs']
    lbl_ts = label['tissue_curve']
    lbl_bd = label['aorta_curve']
    
    # apply scaling 
    if SCALE:
        scale_rand_ts = tf.random.uniform([], 0.5, 2, dtype=tf.float32)
        scale_rand_bd = tf.random.uniform([], 0.8, 2.2, dtype=tf.float32)
    
        inp_ts *= scale_rand_ts
        inp_bd *= scale_rand_bd
        lbl_ts *= scale_rand_ts
        lbl_bd *= scale_rand_bd
    
    # apply noise 
    if NOISE:
        SNR = 20.0
        n_seq = inp_ts.shape[-1]
        noise_var = tf.reduce_sum(lbl_bd**2)/n_seq/(10.0**(SNR/10))
#     print(inp_ts.shape)
    
        inp_ts += tf.random.normal(shape=inp_ts.shape,stddev=noise_var)
        inp_bd += tf.random.normal(shape=inp_ts.shape,stddev=noise_var)
        lbl_ts += tf.random.normal(shape=inp_ts.shape,stddev=noise_var)
        lbl_bd += tf.random.normal(shape=inp_ts.shape,stddev=noise_var)
    
    # apply shifting 
    if SHIFT:
        ts_bundle = tf.stack([inp_ts,lbl_ts],axis=0)
        ts_bundle = tf.image.random_crop(ts_bundle,size=(2,70))
        bd_bundle = tf.stack([inp_bd,lbl_bd],axis=0)
        bd_bundle = tf.image.random_crop(bd_bundle,size=(2,70))

    return {"tissue_ds":ts_bundle[0,:],"aorta_fs":bd_bundle[0,:]},{"tissue_curve":ts_bundle[1,:],"aorta_curve":bd_bundle[1,:]}