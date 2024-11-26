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

    def forward(self, x, y, t, Y=None, Yslice=None):
        '''x, y, t has been broadcasted
        '''
        with torch.no_grad():
            x.to('cuda')
            y.to('cuda')
            t.to('cuda')
            if Y is not None:
                Y.to('cuda')
                if Yslice is None:
                    Yslice = slice(None)
                Y[Yslice].copy_(self._f(x, y, t))
            else:
                return self._f(x, y, t)

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
    output = func(x[:, None, None], y[None, :, None], z[None, None, :])
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
                 method='gauss_legendre_4', flatten=False,
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
            self.__np = {
                'x': int(method[len('gauss_legendre')+1:]) ,
                'y': int(method[len('gauss_legendre')+1:]) ,
                't': int(method[len('gauss_legendre')+1:])
            }
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

        self.__sampling_1d = self._create_sampling_1d()
        self.__w1d = self._create_w1d()
        self.__w_grid_unit_block = Qeff._create_weight_block(self.__w1d)
        self.__u1d = self._create_u1d()
        self.__u_grid_unit_block = Qeff._create_u_block(self.__u1d)

        self.__flatten = flatten

        if isinstance(model, Func):
            self.__func = model
        elif callable(model) and not isinstance(model, torch.nn.Module):
            self.__func = Func(model) # data  model
        elif model is None:
            pass
        else:
            raise NotImplementedError('f must be callable')

    @property
    def func(self):
        return self.__func

    @func.setter
    def func(self, f):
        if isinstance(f, Func):
            self.__func = f
        elif callable(f) and not isinstance(f, torch.nn.Module):
            self.__func = Func(f)
            self.__func.to('cuda')
        else:
            raise NotImplementedError('f must be callable')

    def _create_w1d(self):
      '''1D weights with the size (L, M, N) for x, y, z
      '''
      w1d = {}
      if self.__method == 'cube_corner':
          raise NotImplementedError("cube_corner not implemented.")
      elif 'gauss_legendre' in self.__method:
          for k, n in self.__np.items():
              roots, weights = sp.special.roots_legendre(n)
              w1d[k] = torch.tensor(weights) * self.__gridspacing[k]/2
      else:
          raise NotImplementedError(f"{self.__method} not implemented.")
      return w1d

    def _create_u1d(self):
        '''1D weights for trilinear interpolation
            with the size (L, M, N, 2, 2, 2) for x, y, z
        '''
        u1d = {}
        if self.__method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")

        elif 'gauss_legendre' in self.__method:
            for k, n in self.__np.items():
                n = self.__np[k]
                u1d[k] = torch.zeros([n, 2])
                roots, _ = sp.special.roots_legendre(n)
                roots = torch.tensor(roots)
                u = (roots+1)/2
                u1d[k][:, 0] = 1-u
                u1d[k][:, 1] = u
        else:
          raise NotImplementedError(f"{self.__method} not implemented.")
        return u1d


    @staticmethod
    def _create_weight_block(w1d):
        '''create a weight block'''
        nx = len(w1d['x'])
        ny = len(w1d['y'])
        nt = len(w1d['t'])
        w3d = torch.zeros([nx, ny, nt])
        # better way
        w3d = w1d['x'][:, None, None] * w1d['y'][None, :, None] * w1d['t'][None, None, :]
        # for i in range(nx):
        #     for j in range(ny):
        #         for k in range(nt):
        #             w3d[i,j,k] = w1d['x'][i] * w1d['y'][j] * w1d['t'][k]
        return w3d

    @staticmethod
    def _create_u_block(u1d):
        '''create a weight block for u'''
        w3d = (
            u1d['x'][:, None, None, :, None, None] *
                u1d['y'][None, :, None, None, :, None] *
                u1d['t'][None, None, :, None, None, :]
        )
        return w3d


    def _create_sampling_1d(self):
        '''create grid in 1d with a size of (L, I), (M, ).
        Note:
        1. A uniform spacing of cubes is assumed.
        2. (b-a)/2 of each interval is included in the weight
        '''
        sampling_1d = {}
        if self.__method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in self.__method:

            for k in ['x', 'y', 't']:
                n = self.__np[k]
                roots, weights = sp.special.roots_legendre(n)
                start = self.__space[k][0]
                end = self.__space[k][1]
                npoints = self.__space[k][2]
                corners = torch.linspace(start, end, npoints)

                half_delta = (corners[1:] - corners[:-1])/2. # npoints
                avg = (corners[1:] + corners[:-1])/2. # npoints
                sampling_1d[k] = half_delta[None, :] * roots[:, None] + avg[None, :] # (n, npoints)
        else:
            pass
        return sampling_1d

    def create_qeff_noweight(self, x=None, y=None, t=None, Y=None, Yslice=None):
        '''create Qeff without weights; output is broadcasted; output is not squeezed'''
        if x is None or y is None or t is None:
            x = self.__sampling_1d['x']
            y = self.__sampling_1d['y']
            t = self.__sampling_1d['t']

        if self.__flatten:
            x = x.view(x.shape[0] * x.shape[1])
            y = y.view(y.shape[0] * y.shape[1])
            t = t.view(t.shape[0] * t.shape[1])
            qeff = self.__func(x[:, None, None], y[None, :, None], t[None, None, :])
            return qeff
        else:
            if Y is not None:
                if Yslice is None:
                    Yslice = slice(None)
                self.__func(x[:, None, None, :, None, None],
                            y[None, :, None, None, :, None],
                            t[None, None, :, None, None, :],
                            Y=Y, Yslice=Yslice)
            else:
                qeff = self.__func(x[:, None, None, :, None, None],
                                   y[None, :, None, None, :, None],
                                   t[None, None, :, None, None, :])

    def create_qeff(self):
        '''create Qeff multiplied by weights; output is squeezed'''
        Mx = self.__space['x'][2]-1
        My = self.__space['y'][2]-1
        Mt = self.__space['t'][2]-1
        Nx = self.__np['x']
        Ny = self.__np['y']
        Nt = self.__np['t']
        Ux = self.__u1d['x']
        Uy = self.__u1d['y']
        Ut = self.__u1d['t']
        if self.__flatten:
            qeff = self.create_qeff_noweight()
            qeff = from3Dto6D(qeff, Nx, Ny, Nt, Mx, My, Mt)
            qeff = qeff.permute(3, 4, 5, 0, 1, 2)
            qeff = qeff*self.__w_grid_unit_block.view(1, 1, 1, Nx, Ny, Nt)
            qeff = from6Dto3D(qeff, Mx, My, Mt, Nx, Ny, Nt)
        else:
            qeff = torch.zeros([Nx, Ny, Nt, Mx+2, My+2, Mt+2]) # +2 for padding 0s
            Yslice = (slice(None), slice(None), slice(None),
                slice(1, -1), slice(1, -1), slice(1, -1)) # 1, -1 for excluding pads
            self.create_qeff_noweight(Y=qeff, Yslice=Yslice)
            # reverse u block so that we can cross correlate it with qeff
            # Do I really need reverse? TBD
            kernel = torch.flip(self.__u_grid_unit_block, [3, 4, 5])
            kernel = kernel.view(Nx * Ny * Nt, 1, 2, 2, 2) # out_channel, in_channel/groups, R, S, T
            qeff = qeff.view(1, Nx * Ny * Nt, Mx+2, My+2, Mt+2) # batch, channel, D1, D2, D3
            qeff = torch.nn.functional.conv3d(qeff, kernel, padding='valid',
                                              groups=Nx * Ny * Nt)

            qeff = qeff.view(Nx, Ny, Nt, Mx+1, My+1, Mt+1)
            qeff = qeff*self.__w_grid_unit_block.view(Nx, Ny, Nt, 1, 1, 1)
            qeff = torch.sum(qeff, dim=[0, 1, 2])
            qeff.to('cpu')
        return qeff

if __name__ == '__main__':
    # test_transpose_array()
    test_Func()

    qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
                flatten=True,
                meshgrid=False, method='gauss_legendre_4')
    qeff.func = lambda x, y, t : x**3 * y**3 * t**3
    print(torch.sum(qeff.create_qeff()))

    # test_QModel()

    qeff.func = QModel.create_QModel(X0=(0.4,2.4,3.4), X1=(0.6, 2.6, 3.6),
                              Sigmas=(0.5, 0.5, 0.5))
    print(torch.sum(qeff.create_qeff()))

    qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
                flatten=False,
                meshgrid=False, method='gauss_legendre_4')
    qeff.func = lambda x, y, t : x**3 * y**3 * t**3

    ilinear = lambda x, y, t : x * y * t
    x = np.linspace(0, 1, 11)
    y = np.linspace(2, 3, 11)
    t = np.linspace(3, 4, 11)
    xgrid, ygrid, tgrid = np.meshgrid(x, y, t, indexing='ij')
    I = torch.tensor(ilinear(xgrid, ygrid, tgrid))
    Y = qeff.create_qeff()
    print(torch.sum(Y))
    print(torch.sum(Y * I))
