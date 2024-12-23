import torch
import torch.nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.E = torch.exp(torch.tensor([1.])).item()

import numpy as np
import scipy as sp

import logging
import json

import scipy as sp

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

def test_flip():

    # Create a 2D tensor
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # Flip along (0, 1)
    flip_0_1 = torch.flip(x, dims=(0, 1))

    # Flip along (1, 0)
    flip_1_0 = torch.flip(x, dims=(1, 0))

    # Verify results
    logger.debug('test_flip...')
    logger.debug('Original tensor:')
    logger.debug(x)
    logger.debug("Flip along (0, 1):")
    logger.debug(flip_0_1)
    logger.debug("\nFlip along (1, 0):")
    logger.debug(flip_1_0)

    # Compare results
    logger.debug("\nAre the results equal? {}"\
                 .format(torch.equal(flip_0_1, flip_1_0)))
    assert torch.equal(flip_0_1, flip_1_0)


def test_conv1d(prec=1E-7):
    kernel_size = 3
    X = np.random.rand(2, 2, 2, 4)
    K = np.random.rand(2, 2, 2, kernel_size)
    O = np.zeros([2, 2, 2, 6])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                O[i,j,k] = sp.signal.convolve(X[i,j,k], K[i,j,k],
                                              mode='full')
    # Xtorch = torch.zeros([2, 2, 2, 8])
    # Xtorch[:, :, :, 2:-2] = X
    Xtorch0 = torch.tensor(X, dtype=torch.float32)
    Xtorch1 = torch.nn.functional.pad(
        Xtorch0, (kernel_size-1, kernel_size-1), mode='constant', value=0
    )
    Xtorch2 = torch.zeros(2, 2, 2, 8, dtype=torch.float32)
    Xtorch2[:,:,:,2:-2] = Xtorch0
    Ktorch = torch.tensor(K, dtype=torch.float32)
    Ktorch = torch.flip(Ktorch, dims=(-1,))
    convlayer_pad = torch.nn.Conv1d(in_channels=2*2*2,
                                    out_channels=2*2*2,
                                    kernel_size=kernel_size,
                                    groups=2*2*2,
                                    padding=kernel_size-1,
                                    padding_mode='zeros',
                                    bias=False)
    convlayer_nopad = torch.nn.Conv1d(in_channels=2*2*2,
                                      out_channels=2*2*2,
                                      kernel_size=kernel_size,
                                      groups=2*2*2,
                                      padding='valid',
                                      bias=False)

    convlayer_pad.requires_grad = False
    convlayer_nopad.requires_grad = False
    convlayer_pad.weight = torch.nn.Parameter(Ktorch.view(2*2*2, 1,
                                                          kernel_size),
                                              requires_grad=False)
    convlayer_nopad.weight = torch.nn.Parameter(Ktorch.view(2*2*2, 1,
                                                            kernel_size),
                                                requires_grad=False)

    Otorch0 = convlayer_pad(Xtorch0.view(1, 2*2*2, -1))
    Otorch1 = convlayer_nopad(Xtorch1.view(1, 2*2*2, -1))
    Otorch2 = convlayer_nopad(Xtorch2.view(1, 2*2*2, -1))

    d0 = torch.abs(Otorch0.view(2,2,2,-1) - torch.tensor(O))
    d1 = torch.abs(Otorch1.view(2,2,2,-1) - torch.tensor(O))
    d2 = torch.abs(Otorch2.view(2,2,2,-1) - torch.tensor(O))
    assert torch.all(d0 < prec).item()
    assert torch.all(d1 < prec).item()
    assert torch.all(d2 < prec).item()


def test_conv2d(prec=1E-6):
    input_size = 5, 5
    kernel_size = 3, 3
    output_size = tuple( i + 1*(k-1)
                         for i, k in zip(input_size, kernel_size) )
    padding_size = tuple( k-1 for k in kernel_size )
    X = np.random.rand(2, 2, 2, input_size[0], input_size[1])
    K = np.random.rand(2, 2, 2, kernel_size[0], kernel_size[1])
    O = np.zeros([2, 2, 2, output_size[0], output_size[1]])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                O[i,j,k] = sp.signal.convolve(X[i,j,k], K[i,j,k],
                                              mode='full')
    Xtorch0 = torch.tensor(X, dtype=torch.float32)
    Xtorch1 = torch.nn.functional.pad(
        Xtorch0, (kernel_size[0]-1, kernel_size[0]-1,
                  kernel_size[1]-1, kernel_size[1]-1),
        mode='constant', value=0
    )
    Xtorch2 = torch.zeros(2, 2, 2, output_size[0] + kernel_size[0]-1,
                          output_size[1] + kernel_size[1]-1,
                          dtype=torch.float32)
    Xtorch2[:,:,:, kernel_size[0]-1:1-kernel_size[0],
            kernel_size[1]-1:1-kernel_size[1]] = Xtorch0
    Ktorch = torch.tensor(K, dtype=torch.float32)
    Ktorch = torch.flip(Ktorch, dims=(-2,-1))
    convlayer_pad = torch.nn.Conv2d(in_channels=2*2*2,
                                    out_channels=2*2*2,
                                    kernel_size=kernel_size,
                                    groups=2*2*2,
                                    padding=padding_size,
                                    padding_mode='zeros',
                                    bias=False)
    convlayer_nopad = torch.nn.Conv2d(in_channels=2*2*2,
                                      out_channels=2*2*2,
                                      kernel_size=kernel_size,
                                      groups=2*2*2,
                                      padding='valid',
                                      bias=False)

    convlayer_pad.requires_grad = False
    convlayer_nopad.requires_grad = False
    convlayer_pad.weight = torch.nn.Parameter(Ktorch.view(2*2*2, 1,
                                                          kernel_size[0],
                                                          kernel_size[1]),
                                              requires_grad=False)
    convlayer_nopad.weight = torch.nn.Parameter(Ktorch.view(2*2*2, 1,
                                                            kernel_size[0],
                                                            kernel_size[1]
                                                            ),
                                                requires_grad=False)

    Otorch0 = convlayer_pad(Xtorch0.view(1, 2*2*2, input_size[0],
                                         input_size[1]))
    Otorch1 = convlayer_nopad(Xtorch1.view(1, 2*2*2,
                                           input_size[0]+2*padding_size[0],
                                           input_size[1]+2*padding_size[1]
                                           ))
    Otorch2 = convlayer_nopad(Xtorch2.view(1, 2*2*2,
                                           input_size[0]+2*padding_size[0],
                                           input_size[1]+2*padding_size[1]
                                           ))

    d0 = torch.abs(Otorch0.view(2,2,2,output_size[0],output_size[1])
                   - torch.tensor(O))
    d1 = torch.abs(Otorch1.view(2,2,2,output_size[0],output_size[1])
                   - torch.tensor(O))
    d2 = torch.abs(Otorch2.view(2,2,2,output_size[0],output_size[1])
                   - torch.tensor(O))

    assert torch.all(d0 < prec).item()
    assert torch.all(d1 < prec).item()
    assert torch.all(d2 < prec).item()

def test_conv3d(prec=2E-6):
    input_size = 5, 6, 7
    kernel_size = 3, 3, 3
    output_size = tuple( i + 1*(k-1)
                         for i, k in zip(input_size, kernel_size) )
    padding_size = tuple( k-1 for k in kernel_size )
    X = np.random.rand(2, 2, 2, input_size[0], input_size[1],
                       input_size[2])
    K = np.random.rand(2, 2, 2, kernel_size[0], kernel_size[1],
                       kernel_size[2])
    O = np.zeros([2, 2, 2, output_size[0], output_size[1],
                  output_size[2]])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                O[i,j,k] = sp.signal.convolve(X[i,j,k], K[i,j,k],
                                              mode='full')
    Xtorch0 = torch.tensor(X, dtype=torch.float32)
    Xtorch1 = torch.nn.functional.pad(
        Xtorch0, (kernel_size[0]-1, kernel_size[0]-1,
                  kernel_size[1]-1, kernel_size[1]-1,
                  kernel_size[2]-1, kernel_size[2]-1),
        mode='constant', value=0
    )
    Xtorch2 = torch.zeros(2, 2, 2, output_size[0] + kernel_size[0]-1,
                          output_size[1] + kernel_size[1]-1,
                          output_size[2] + kernel_size[2]-1,
                          dtype=torch.float32)
    Xtorch2[:,:,:, kernel_size[0]-1:1-kernel_size[0],
            kernel_size[1]-1:1-kernel_size[1],
            kernel_size[2]-1:1-kernel_size[2]] = Xtorch0
    Ktorch = torch.tensor(K, dtype=torch.float32)
    Ktorch = torch.flip(Ktorch, dims=(-3,-2,-1))
    convlayer_pad = torch.nn.Conv3d(in_channels=2*2*2,
                                    out_channels=2*2*2,
                                    kernel_size=kernel_size,
                                    groups=2*2*2,
                                    padding=padding_size,
                                    padding_mode='zeros',
                                    bias=False)
    convlayer_nopad = torch.nn.Conv3d(in_channels=2*2*2,
                                      out_channels=2*2*2,
                                      kernel_size=kernel_size,
                                      groups=2*2*2,
                                      padding='valid',
                                      bias=False)

    convlayer_pad.requires_grad = False
    convlayer_nopad.requires_grad = False
    convlayer_pad.weight = torch.nn.Parameter(Ktorch.view(2*2*2, 1,
                                                          kernel_size[0],
                                                          kernel_size[1],
                                                          kernel_size[2]),
                                              requires_grad=False)
    convlayer_nopad.weight = torch.nn.Parameter(Ktorch.view(2*2*2, 1,
                                                            kernel_size[0],
                                                            kernel_size[1],
                                                            kernel_size[2]
                                                            ),
                                                requires_grad=False)

    Otorch0 = convlayer_pad(Xtorch0.view(1, 2*2*2, input_size[0],
                                         input_size[1],
                                         input_size[2]))
    Otorch1 = convlayer_nopad(Xtorch1.view(1, 2*2*2,
                                           input_size[0]+2*padding_size[0],
                                           input_size[1]+2*padding_size[1],
                                           input_size[2]+2*padding_size[2]
                                           ))
    Otorch2 = convlayer_nopad(Xtorch2.view(1, 2*2*2,
                                           input_size[0]+2*padding_size[0],
                                           input_size[1]+2*padding_size[1],
                                           input_size[2]+2*padding_size[2]
                                           ))

    d0 = torch.abs(Otorch0.view(2,2,2,output_size[0],output_size[1],
                                output_size[2])
                   - torch.tensor(O))
    d1 = torch.abs(Otorch1.view(2,2,2,output_size[0],output_size[1],
                                output_size[2])
                   - torch.tensor(O))
    d2 = torch.abs(Otorch2.view(2,2,2,output_size[0],output_size[1],
                                output_size[2])
                   - torch.tensor(O))
    assert torch.all(d0 < prec).item()
    assert torch.all(d1 < prec).item()
    assert torch.all(d2 < prec).item()


def from3Dto6D(X, Mx, My, Mz, Nx, Ny, Nz):
    '''[i * Nx + l, j * Nm + m, k * Nt + n]
    --> [i][j][k][l][m][n]
    for i in Mx, j in My, k in Mz, l in Nx, m in Ny, n in Nz
    '''
    y = X.reshape(Mx, Nx, My, Ny, Mz, Nz)
    y = torch.permute(y, (0, 2, 4, 1, 3, 5))
    return y


def from6Dto3D(X, Mx, My, Mz, Nx, Ny, Nz):
    '''[i][j][k][l][m][n]
    --> [i * Nx + l, j * Nm + m, k * Nt + n]
    for i in Mx, j in My, k in Mz, l in Nx, m in Ny, n in Nz
    '''
    y = torch.permute(X, (0, 3, 1, 4, 2, 5))
    y = y.reshape(Mx*Nx, My*Ny, Mz*Nz)
    return y

def test_transpose_array():
    M = 3
    N = 2
    x = torch.arange(0, N**3*M**3).reshape(M, M, M, N, N, N)
    # y = torch.permute(x, (0,3, 1,4, 2,5)).reshape(M*N, M*N, M*N)
    y = from6Dto3D(x, M, M, M, N, N, N)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    for m in range(x.shape[4]):
                        for n in range(x.shape[5]):
                            d = x[i,j,k,l,m,n] - y[x.shape[3]*i+l,
                                                   x.shape[4]*j+m,
                                                   x.shape[5]*k+n]
                            if torch.abs(d) > 1E-6:
                                print("distance > 1E-6", d, i, j, k, l, m, n)
                            assert torch.abs(d) < 1E-6
    x = from3Dto6D(y, M, M, M, N, N, N)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    for m in range(x.shape[4]):
                        for n in range(x.shape[5]):
                            d = x[i,j,k,l,m,n] - y[x.shape[3]*i+l,
                                                   x.shape[4]*j+m,
                                                   x.shape[5]*k+n]
                            if torch.abs(d) > 1E-6:
                                print("distance > 1E-6", d)
                            assert torch.abs(d) < 1E-6

if __name__ == '__main__':

    test_transpose_array()

    test_flip()
    test_conv1d(prec=1E-6)
    test_conv2d(prec=1E-6)
    test_conv3d(prec=2E-6)
