"""
simulation functions
"""
import numpy as np 
import math

import tensorflow as tf
def cut_bound(vp,ve,ktr,vpt=0.01,vpr=0.05,vet=0.2,ver=0.5,ktrt=0.002,ktrr=0.01):
    if vp>0.05:
        vp = 0.05
    if vp<0.01:
        vp = 0.01
    if ve>0.5:
        ve = 0.5
    if ve<0.2:
        ve=0.2
    if ktr>0.01:
        ktr = 0.01
    if ktr<0.002:
        ktr = 0.002
    return vp,ve,ktr
def con_2cxm_constant(A,t,alpha1,alpha2,tau1,tau2):
    return A*(alpha1*np.exp(-t/tau1)+alpha2*(np.exp(-t/tau2)))
def decomp_mat(alpha, n, b):
# alpha: coefficient for exp
# n: size of matrix
# b: vector for convolution
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j>=i:
                M[i,j] = b[i]*np.exp(-alpha*(j-i))
    return M 
def con_tofts_matlab(c_pl_used,vp,ve,ktr,delta_t=1.4):
    n=len(c_pl_used)
    return vp*c_pl_used+ktr*np.sum(decomp_mat(ktr/ve*delta_t, n, c_pl_used),axis=0)*delta_t
def con_oritofts_matlab(c_pl_used,ve,ktr,delta_t=1.4):
    n=len(c_pl_used)
    return ktr*np.sum(decomp_mat(ktr/ve*delta_t, n, c_pl_used),axis=0)*delta_t
def con_oritofts_matlab_kep(c_pl_used,kep,ktr,delta_t=1.4):
    n=len(c_pl_used)
    return ktr*np.sum(decomp_mat(kep*delta_t, n, c_pl_used),axis=0)*delta_t

def con_patlak_tf_fit(c,vp,ktr,padding=69,del_t=1.4):
    """
    input: c (N_pixel,N_temporal),vp,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    ktr = tf.cast(ktr,tf.float32)
    vp = tf.cast(vp,tf.float32)
    c = tf.cast(c,tf.float32)
    n = tf.constant(range(c.shape[-1]))
    p1 = tf.reshape(vp,(-1,1))*c
    length = c.shape[-1]
    padding = [[0,0],[padding,0],[0,0]]
    c_new=p1+ktr*tf.reshape(tf.math.cumsum(c,axis=-1),(1,-1))*del_t
    return c_new.numpy().flatten()
def fun_oritofts_LLSQ(c_p,c_t,ktr,kep,reg=1,del_t=1.4):
    """
    function to be minimized in scipy.optimize.least_squares
    """
    ktr = tf.cast(ktr,tf.float32)
    kep = tf.cast(kep,tf.float32)
    c_p = tf.cast(c_p,tf.float32)
    c_t = tf.cast(c_t,tf.float32)
    lhs = c_t 
    rhs = ktr*tf.reshape(tf.math.cumsum(c_p,axis=-1),(1,-1))*del_t - kep*tf.reshape(tf.math.cumsum(c_p,axis=-1),(1,-1))*del_t
    return reg*(lhs - rhs).numpy().flatten()
def con_patlak_tf_fit_360(c,vp,ktr,del_t=1.4):
    """
    input: c (N_pixel,N_temporal),vp,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    padding = c.shape[-1]-1
    print(padding)
    ktr = tf.cast(ktr,tf.float32)
    vp = tf.cast(vp,tf.float32)
    c = tf.cast(c,tf.float32)
    n = tf.constant(range(c.shape[-1]))
    print(padding)
    p1 = tf.reshape(vp,(-1,1))*c
    length = c.shape[-1]
    padding = [[0,0],[padding,0],[0,0]]
    c_new=p1+ktr*tf.reshape(tf.math.cumsum(c,axis=-1),(1,-1))*del_t
    return c_new.numpy().flatten()
def con_tofts(c,vp,ve,ktr,del_t=1.4):
    """
    input: c (N_pixel,N_temporal),vp,ve,ktr(N_pixel,1)
    output: c_new (N_pixel,N_temporal)
    """
    n = np.array(range(c.shape[-1]))
    ktr = ktr/60
#     c_new = (vp.reshape(-1,1)*c+
#              (ktr.reshape(-1,1)*np.exp((-ktr/ve).reshape(-1,1).dot(n.reshape(1,-1))*del_t))*np.cumsum(c*len_tofts(ktr,ve,n,del_t),axis=-1)*del_t)
    inp = np.hstack((c,len_tofts(-ktr,ve,n,del_t)))
    length = c.shape[-1]
    c_new = vp.reshape(-1,1)*c+ktr.reshape(-1,1)*np.apply_along_axis(lambda m: np.convolve(m[:length],m[length:],  mode='full')[:length], axis=-1, arr=inp)*del_t
    return c_new
def len_tofts (ktr,ve,n,del_t):
    return np.exp(np.reshape((ktr/ve),(-1,1)).dot(np.reshape(n,(1,-1)))*del_t)

def gen_con_bd(N=360,r_st_ran=(90,100),r_t_ran=(10,16),A_ran=(3,5),tau1_ran=(20,40),tau2_ran=(800,1200),
           alpha1=0.5,alpha2=0.5): 
    """
    Inputs:
    - N: total time point, default 360
    Random variables:
    - r_st: rising start point
    - r_t : rising duration 
    - A: random scale 
    - tau1, tau2, alpha1, alpha2: parameters in two compartment exchange model
    Outputs:
    - A random generated AIF signal(SI) with shape(N,1)
    """
    A = np.random.uniform(low=A_ran[0],high=A_ran[1])
    r_st = np.random.randint(r_st_ran[0],r_st_ran[1])
    r_t = np.random.randint(r_t_ran[0],r_t_ran[1])
    tau1 = np.random.uniform(low=tau1_ran[0],high=tau1_ran[1])
    tau2 = np.random.uniform(low=tau2_ran[0],high=tau2_ran[1])
    t = np.arange(start=r_st+r_t,stop=N)-(r_st+r_t)+1
    c_1 = np.zeros(r_st) #first stage: pre-injection
    c_2 = A*np.linspace(0,1,num=r_t,axis=-1) #second stage: rising
    c_3 = con_2cxm_constant(A,t,alpha1,alpha2,tau1,tau2) #third stage: perfusion
    return np.concatenate((c_1,c_2,c_3),axis=-1), c_1,c_2,c_3

# def get_random_kinetic_uniform(Ft=1.0/60,Fr=0.1/60,vpt=0.2,vpr=0.05,vet=0.2,ver=0.4,PSt=0.08/60,PSr=0.03/60,size=None):
#     """
#     Two compartment exchange model
#     """
#     F = np.random.uniform(low=np.minimum(Ft,Fr),high=np.maximum(Ft,Fr),size=size)
#     vp = np.random.uniform(low=np.minimum(vpt,vpr),high=np.maximum(vpt,vpr),size=size)    
#     ve = np.random.uniform(low=np.minimum(vet,ver),high=np.maximum(vet,ver),size=size)
#     PS = np.random.uniform(low=np.minimum(PSt,PSr),high=np.maximum(PSt,PSr),size=size)
#     return F,vp,ve,PS
# def get_interm_variable(F,vp,ve,PS):
#     """
#     Two compartment exchange model
#     """
#     Te = ve/PS
#     T = (vp+ve)/F
#     Tc = vp/F
#     alpha = ( (T+Te)+np.sqrt((T+Te)**2-4*Tc*Te) )/( 2*Tc*Te )
#     beta  = ( (T+Te)-np.sqrt((T+Te)**2-4*Tc*Te) )/( 2*Tc*Te )
#     A = PS * ( vp*Te*alpha - vp - ve )/((alpha-beta) * vp * ve)
#     ktr = PS * F/(PS+F)
#     return A,ktr,alpha,beta
def get_random_kinetic_uniform(vpt=0.01,vpr=0.05,vet=0.2,ver=0.5,ktrt=0.002,ktrr=0.01,size=None): #vet 0.2 ver 0.4
    """
    Tofts model
    """
    vp = np.random.uniform(low=np.minimum(vpt,vpr),high=np.maximum(vpt,vpr),size=size)    
    ve = np.random.uniform(low=np.minimum(vet,ver),high=np.maximum(vet,ver),size=size)
    ktr = np.random.uniform(low=np.minimum(ktrt,ktrr),high=np.maximum(ktrt,ktrr),size=size)
    return vp,ve,ktr

def get_random_R1(T1t=1.600,T1r=0.800,T1bd_min=1.500,T1bd_max=1.600,size=None):# blood 1.4 ~ 2 tissue 1.5 0.8
    T1 = np.random.uniform(low=np.minimum(T1t,T1r),high=np.maximum(T1t,T1r),size=size)
    R1 = 1/T1
    T1_bd = np.random.uniform(low=T1bd_min,high=T1bd_max,size=size)
    R1_bd = 1/T1_bd 
    return R1,R1_bd

def gen_dyna_R1(R1,R1_bd,con_bd,con_pan,r=3.2,N_ds=4):
    # generate dynamic R1,
    # then downsample concentration curve to 4 times long squence
    R1_bd_ = R1_bd+r*con_bd
    R1_pan_ = R1+r*con_pan
    length = len(con_bd)
    R1_bd__ = np.interp(np.arange(0,length*N_ds,1),np.arange(0,length*N_ds,N_ds),R1_bd_)
    R1_pan__ = np.interp(np.arange(0,length*N_ds,1),np.arange(0,length*N_ds,N_ds),R1_pan_)
    return R1_pan__, R1_bd__
def e1(R1,ES=5.6e-3):
    # small TR T1 decay
    return np.exp(-ES*R1)
def Mss(R1,alpha=10*math.pi/180.0):
    return(1-e1(R1))/(1-np.cos(alpha)*e1(R1))*np.sin(alpha)
def S_mul_task(R1,Nseg=84,alpha=10*math.pi/180.0):
    N=np.arange(Nseg).reshape(-1,1)
    S = (Mss(R1=R1,alpha=alpha)*(1-((e1(R1)*np.cos(alpha))**N))).transpose()
    return S
def gen_noise(S,snr=20):
    noise_var = np.sum(S**2)/np.prod(S.shape)/(10**(snr/10))
    noise = np.random.normal(0, np.sqrt(noise_var), S.shape)
    Sn = S + noise
    return Sn
def gen_U0(S_bd,S_pan,cPinv):
    """
    Processing blood and tissue signal to deep-learning-able data format
    Inputs:
    - S_bd: multitasking signal of blood, shape (DCE_DIM,SR_DIM) typically (360,84)
    - S_pan: multitasking signal of tissue, same shape of S_bd
    - cPinv: pseudo-inverse of curvePhi: T1 basis function, generated in matlab, loaded, shape(SR_DIM,cL), typically (84,5)
    Outpus:
    - U0: Input of network. shape (DCE_DIM, 2*cL)
    """
    
    """
    Step1: Shrink SR dimension to cL
    """
    S_bd_0 = S_bd.dot(cPinv)
    S_pan_0 = S_pan.dot(cPinv)
    """
    Step2: Dealing with complex value
    Note in simulation we all have real value, so neglect this step
    """
    """
    Step3: Scaling (pixel-wise)
    Note in vivo data we use all-image to normalize, so slightly different
    """
    cw = np.amax(S_pan_0)
    scale = np.array([2,50,75,180,210])
    U0 = np.concatenate((S_bd_0*scale,S_pan_0*scale), axis=-1)/cw # scale 2.06590759  64.79525627  74.14727259 183.44910667 210.82790066
    return U0


