import scipy
import numpy as np
import tensorflow as tf

def conv1d(a, b):
    '''
    always assume a size > b size
    '''
    na = a.shape[0]
    nb = b.shape[0]
    c = np.zeros(na+nb-1)
    nc = c.shape[0]
    for k in range(nc):
        if k < nb:
            x1 = a[0:k+1]
            x2 = b[0:k+1][::-1]

        if k >= nb and k < na:
            x1 = a[k-nb+1:k+1]
            x2 = b[::-1]

        if k >= na:
            x1 = a[k-nb+1:na]
            x2 = b[:-(na+nb-k):-1]

        c[k] = np.sum(x1*x2)

    return c

def conv1d_padzero(a, b):
    '''always assume a size > b size
    '''
    na = a.shape[0]
    nb = b.shape[0]

    c = np.zeros(na+nb-1)
    nc = c.shape[0]

    acopy = np.concatenate([np.zeros(nb-1),
                            a, np.zeros(nb-1)])
    for k in range(nc):
        c[k] = np.sum(acopy[k:k+nb] * b[::-1])

    return c

# https://stackoverflow.com/a/60289440
# scipy.signal.convolve
# numpy convolve
def conv1d_np(a, b):
    return np.convolve(a, b)

def conv1d_scipy(a, b):
    # method 1; not desired dimension
    # bcopy = np.zeros(a.shape)
    # bcopy[:b.shape[0]] = b
    # c = scipy.signal.convolve(a, bcopy)

    # method 2; not desired dimension
    # c = scipy.signal.convolve(a, b, mode='valid')

    # method 3; not desired dimension
    # acopy = np.zeros(a.shape[0]+b.shape[0]-1)
    # acopy[:a.shape[0]] = a
    # c = scipy.signal.convolve(acopy, b, mode='valid')

    # method 4; not desired dimension
    # c = scipy.signal.convolve(a, b, mode='same')

    # method 5; not desired dimension
    # acopy = np.zeros(a.shape[0]+b.shape[0]-1)
    # acopy[:a.shape[0]] = a
    # c = scipy.signal.convolve(acopy, b, mode='same')

    # method 6;
    c = scipy.signal.convolve(a, b, mode='full')


    return c

def conv2d_padzero(a, b):
    '''always assume a size > bsize in each dimension
    '''
    na0, na1 = a.shape
    nb0, nb1 = b.shape
    c = np.zeros([na0+2*nb0, na1+2*nb1])
    c[nb0-1:na0+nb0-1, nb1-1:na1+nb1-1] = a

    d = np.zeros([na0+nb0-1,na1+nb1-1])
    
    nc0, nc1 = c.shape
    nd0, nd1 = d.shape
    
    for i in range(nd0):
        for j in range(nd1):
            d[i,j] = np.sum(c[i:i+nb0,j:j+nb1] * b[::-1,::-1])
            
    return d

def conv2d_scipy(a, b):
    # method 1
    # na0, na1 = a.shape
    # nb0, nb1 = b.shape
    # c = np.zeros([na0+nb0, na1+nb1])
    # c[nb0-1:na0+nb0-1, nb1-1:na1+nb1-1] = a
    # d = scipy.signal.convolve2d(c, b, mode='valid')

    # method 2
    # d = scipy.signal.convolve2d(a, b)

    # method 3
    d = scipy.signal.convolve(a, b)

    return d

def conv3d_padzero(a, b):
    na0, na1, na2 = a.shape
    nb0, nb1, nb2 = b.shape
    c = np.zeros([na0+2*nb0-2, na1+2*nb1-2, na2+2*nb2-2])
    c[nb0-1:nb0-1+na0,nb1-1:nb1-1+na1,nb2-1:nb2-1+na2] = a

    d = np.zeros([na0+nb0-1, na1+nb1-1, na2+nb2-1])
    nd0, nd1, nd2 = d.shape
    for i in range(nd0):
        for j in range(nd1):
            for k in range(nd2):
                d[i,j,k] = np.sum(c[i:i+nb0,j:j+nb1,k:k+nb2]
                                  *b[::-1,::-1,::-1])
    return d

def conv3d_scipy(a, b):
    c = scipy.signal.convolve(a, b)
    return c

def conv3d_scipy_valid(a, b):
    na0, na1, na2 = a.shape
    nb0, nb1, nb2 = b.shape
    c = np.zeros([na0+2*nb0-2, na1+2*nb1-2, na2+2*nb2-2])
    c[nb0-1:nb0-1+na0,nb1-1:nb1-1+na1,nb2-1:nb2-1+na2] = a
    d = scipy.signal.convolve(c, b, mode='valid')

    return d

def conv3d_tf(a, b):
    
    X = a
    F = b[::-1,::-1,::-1]
    # Convert X and F to TensorFlow tensors and reshape for 3D convolution
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    # Shape (batch, depth, height, width, channels)
    X_tf = tf.reshape(X_tf, (1, 3, 3, 3, 1))  

    # Shape (filter_depth, filter_height, filter_width,
    # in_channels, out_channels)
    F_tf = tf.convert_to_tensor(F, dtype=tf.float32)
    F_tf = tf.reshape(F_tf, (2, 2, 2, 1, 1))

    # Calculate the necessary padding for "full" convolution
    padding_d = F.shape[0] - 1  # Depth padding
    padding_h = F.shape[1] - 1  # Height padding
    padding_w = F.shape[2] - 1  # Width padding

    # Apply padding to the input tensor
    X_tf_padded = tf.pad(X_tf, [[0, 0],  # No padding for batch dimension
                                [padding_d, padding_d],  # Depth padding
                                [padding_h, padding_h],  # Height padding
                                [padding_w, padding_w],  # Width padding
                                [0, 0]])  # No padding for channels

    # Perform 3D convolution in TensorFlow with "VALID" padding
    y_tf = tf.nn.conv3d(X_tf_padded, F_tf,
                        strides=[1, 1, 1, 1, 1], padding="VALID")

    # Remove extra dimensions to get a 3D result
    y_tf_squeezed = tf.squeeze(y_tf)

    return y_tf_squeezed

def rebin_by_sum(X, k):
    '''
    X.shape must be (, km * M, kn * N)
    k: integer or (km, kn)
    '''
    if isinstance(k, int):
        km = k
        kn = k
    else:
        km, kn = k
    Mk = X.shape[-2]
    Nk = X.shape[-1]
    M = Mk // km
    N = Nk // kn
    axm = len(X.shape) - 2 + 1
    axn = len(X.shape) - 2 + 3
    y = X.reshape(-1, M, km, N, kn).sum(axis=(axm, axn))
    return y

def reorder_3rd(X):
    return np.transpose(X, axes=(2, 0, 1))

if __name__ == '__main__':
    X = np.array([1, 2, 3])
    F = np.array([2, 3])
    print('input X', X)
    print('input F', F)

    print('conv1d', conv1d(X, F))
    print('conv1d_padzero', conv1d_padzero(X, F))
    print('np', np.convolve(X, F))
    print('scipy', conv1d_scipy(X, F))

    # https://stackoverflow.com/a/51865516
    X = np.array([[1, 2, 3], [4, 5, 6]])
    F = np.array([[10, 20], [30, 40]])
    print('input X', X)
    print('input F', F)
    print('conv2d_padzero', conv2d_padzero(X, F))
    print('conv2d_scipy', conv2d_scipy(X, F))

    X = np.arange(1, 28).reshape([3, 3, 3])
    F = np.arange(1, 9).reshape([2, 2, 2])
    print('input X', X)
    print('input F', F)
    print('conv3d_padzero', conv3d_padzero(X, F))
    print('conv3d_scipy', conv3d_scipy(X, F))
    print('conv3d_scipy_valid', conv3d_scipy(X, F))
    print('conv3d_tf', conv3d_tf(X, F))

    X = np.arange(0, 4*4*2).reshape(4, 4, 2)
    print('input X', X)
    X = reorder_3rd(X)
    print('after transpose')
    print('input X[0]', X[0])
    print('intput X[1]', X[1])
    print('rebin by sum', rebin_by_sum(X, (2, 2)))
