import torch
import torch.nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.E = torch.exp(torch.tensor([1.])).item()

import numpy as np
import scipy as sp

import logging
import json

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

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

class Func(torch.nn.Module):
    def __init__(self, f=None):
        super(Func, self).__init__()
        if isinstance(f, str):
            raise NotImplementedError("TBD, str to predefined function.")
        elif callable(f):
            self._f = f
        elif f is None:
            raise NotImplementedError("TBD, add a default test function.")
        else:
            raise NotImplemented(f"Not supported for type {type(f)}")
    def forward(self, x, y, t):
        '''x, y, t are 1-d arrays before broadcasting
        '''
        with torch.no_grad():
            return self._f(x[:, None, None],
                           y[None, :, None], t[None, None, :])

    @property
    def f(self):
        return self._f

class QModel():
    def __new__(cls, *args, **kwargs):
        raise TypeError("This class cannot be instantiated.")
    @staticmethod
    def GaussConvLine(x, y, t, X0, X1, Sigmas):
        x0, y0, t0 = torch.tensor(X0, dtype=torch.float32)
        x1, y1, t1 = torch.tensor(X1, dtype=torch.float32)
        sx, sy, st = torch.tensor(Sigmas, dtype=torch.float32)

        deltaSquare = (
            st**2 * sy**2 * (x0 - x1)**2
            + sx**2 * (sy**2 * (t0 - t1)**2 + st**2 * (y0 - y1)**2)
        )

        deltaSquareSqrt = np.sqrt(deltaSquare)

        argA1 = (
            st**2 * sy**2 * (x - x0)*(x0 - x1) +
            sx**2*(sy**2 *(t - t0)*(t0 - t1) +
                   st**2 * (y - y0)*(y0 - y1))
        )
        argA2 = (
            st**2 * sy**2 * (x - x1)*(x0 - x1) +
            sx**2 * (sy**2 * (t - t1)*(t0 - t1) +
                     st**2 * (y - y1)*(y0 - y1))
        )
        argB = (
            sy**2 * torch.pow(t1*(x0 - x) + t0*(x - x1) + t*(x1 - x0), 2) +
            sx**2 * torch.pow(t1*(y0 - y) + t*(y1 - y0) + t0*(y - y1), 2) +
            st**2 * torch.pow(x0*(y - y1) + x1*(y0 - y) + x*(y1 - y0), 2)
        )
        output = (
            -1* torch.exp(-0.5*argB/deltaSquare)/4/np.pi/deltaSquareSqrt * (
                torch.erf(argA1/np.sqrt(2)/deltaSquareSqrt/sx/sy/st) -
                torch.erf(argA2/np.sqrt(2)/deltaSquareSqrt/sx/sy/st)
            )
        )

        return output


    @staticmethod
    def create_QModel(X0, X1, Sigmas):
        f = lambda x, y, t : QModel.GaussConvLine(x, y, t, X0, X1, Sigmas)
        return f


def test_Func():
    f = lambda x, y, t : x**2 + y**2 + t**2
    func = Func(f)
    x = torch.linspace(0, 1, 2).to('cuda')
    y = torch.linspace(1, 2, 2).to('cuda')
    z = torch.linspace(3, 4, 2).to('cuda')
    func = func.to('cuda')
    output = func(x, y, z)
    for i, vi in zip([0,1], [0,1]):
        for j, vj in zip([0,1], [1,2]):
            for k, vk in zip([0,1], [3,4]):
                d = output[i,j,k] - f(vi, vj, vk)
                assert torch.abs(d) < 1E-6


def test_QModel():
    func = QModel.create_QModel(X0=(0.4,2.4,3.4), X1=(0.6, 2.6, 3.6),
                              Sigmas=(0.05, 0.05, 0.05))
    func = Func(func)

    x = torch.linspace(0.2, 0.8, 5, dtype=torch.float64)
    y = torch.linspace(2.2, 2.8, 5, dtype=torch.float64)
    t = torch.linspace(3.2, 3.8, 5, dtype=torch.float64)
    testq = func(x,y,t)
    with open('exact.json') as f:
        exact = json.load(f)
    for i in range(testq.shape[0]):
        for j in range(testq.shape[1]):
            for k in range(testq.shape[2]):
                d = np.abs(testq[i,j,k].item()-exact[i][j][k])
                if d > 1E-5:
                    print('difference > 1E-5', i, j, k,
                          x[i].item(), y[j].item(), t[k].item(),
                          testq[i,j,k].item(),exact[i][j][k],  d)


class Qeff():
    '''Qeff'''
    def __init__(self, xspace, yspace, tspace, meshgrid=False,
                 method='gauss_legendre_4', initialize=False,
                 model=None):
        '''xspace, yspcae, tspace are tuples of three numbers
        defining the range and number of grid points along one dimension
        of the charge space.
        Note: The end point is always included.
        Input (1, 2, 3) will yield ([1, 1.5, 2].
        '''
        self._logger = logger.getChild('class.Qeff')
        self.__space = {
            'x' : xspace,
            'y' : yspace,
            't' : tspace
        }
        if xspace is None or yspace is None or tspace is None:
            raise NotImplementedError("Range of charge space is required.")

        self.__gridshape = tuple(self.__space[k][2]
                                 for k in ['x', 'y', 't'])
        # self.__qeffshape = self.__gridshape
        self.__gridspacing = {
            k : (v[1]-v[0])/(v[2]-1) for k, v in self.__space.items()
        }

        self.__meshgrid = meshgrid

        if meshgrid is True:
            raise NotImplementedError('Does not support to use meshgrid yet.')

        if method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in method[0:len('gauss_legendre')]:
            self.__np = int(method[len('gauss_legendre')+1:])
        else:
            msg = (
                f"Method {method} not supported. "
                "Available options are cube_corner, gauss_legendre_n, "
                "where n means n-point Gauss-Legendre Quadrature. "
                "For instance, gauss_legendre_2 means two-point "
                "Gauss-Legendre Quadrature."
            )
            raise NotImplementedError(msg)

        self.__method = method

        self.__grid_1d, _, w_grid_1d_unit = self._create_grid1d()
        self.__w_grid_unit_block = Qeff._create_weight_block(w_grid_1d_unit)

        self.__func = model # data  model

    @property
    def func(self):
        return self.__func

    @func.setter
    def func(self, f):
        if callable(f):
            self.__func = Func(f)
            self.__func.to('cuda')
        else:
            raise NotImplementedError('f must be callable')

    @staticmethod
    def _create_weight_block(w1d):
        '''create a weight block'''
        nx = len(w1d['x'])
        ny = len(w1d['y'])
        nt = len(w1d['t'])
        w3d = torch.zeros([nx, ny, nt])
        for i in range(nx):
            for j in range(ny):
                for k in range(nt):
                    w3d[i,j,k] = w1d['x'][i] * w1d['y'][j] * w1d['t'][k]
        return w3d


    def _create_grid1d(self):
        '''create grid and corresponding weights in 1d.
        weight grid a long 1d is useless
        weight grid 1d in one cube is useful
        Note:
        1. A uniform spacing of cubes is assumed.
        2. (b-a)/2 of each interval is included in the weight
        '''
        grid_1d = {}
        w_grid_1d = None
        w_grid_1d_unit = {}
        if self.__method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in self.__method:
            n = self.__np
            roots, weights = sp.special.roots_legendre(n)
            for k in ['x', 'y', 't']:
                corners = torch.linspace(self.__space[k][0],
                                         self.__space[k][1], self.__space[k][2])
                # to be optimized
                grid_1d[k] = torch.concatenate(list(
                    (b-a)/2 * roots + (b+a)/2
                    for (a, b) in zip(corners[:-1], corners[1:])
                ))
                # assume equal spacing
                step = torch.abs(corners[1] - corners[0])

                w_grid_1d_unit[k] = torch.tensor(weights)*step/2.
        else:
            pass
        return grid_1d, w_grid_1d, w_grid_1d_unit

    def create_qeff_noweight(self):
        '''create Qeff without weights; output is 3D'''
        qeff = self.__func(self.__grid_1d['x'], self.__grid_1d['y'],
                           self.__grid_1d['t'])
        return qeff

    def create_qeff(self):
        '''create Qeff multiplied by weights'''
        qeff = self.create_qeff_noweight()
        Mx = self.__space['x'][2]-1
        My = self.__space['y'][2]-1
        Mt = self.__space['t'][2]-1
        Nx = self.__np
        Ny = self.__np
        Nt = self.__np
        qeff = from3Dto6D(qeff, Mx, My, Mt, Nx, Ny, Nt)
        qeff = qeff*self.__w_grid_unit_block.view(1, 1, 1, Nx, Ny, Nt)
        qeff = from6Dto3D(qeff, Mx, My, Mt, Nx, Ny, Nt)
        return qeff

if __name__ == '__main__':
    test_transpose_array()
    test_Func()
    # qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
    #             meshgrid=True, method='gauss_legendre_4')
    qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
                meshgrid=False, method='gauss_legendre_4')
    qeff.func = lambda x, y, t : x**3 * y**3 * t**3
    print(torch.sum(qeff.create_qeff()))

    test_QModel()

    qeff.func = QModel.create_QModel(X0=(0.4,2.4,3.4), X1=(0.6, 2.6, 3.6),
                              Sigmas=(0.05, 0.05, 0.05))
    print(torch.sum(qeff.create_qeff()))
