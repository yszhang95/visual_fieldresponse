from toy_gpu2 import QModel, Qeff

from scipy import integrate
import numpy as np

import torch

import logging

from functools import partial

logger = logging.getLogger(__name__)

def correct_flat(iarray_flat, Nx, Ny, Nt, Mx, My, Mt):
    '''Mx, My, Mt are Npoints GL method
    '''
    iarray_flat = iarray_flat.view(Mx, Nx,
                                   My, Nx, Mt, Nt)
    iarray_flat = iarray_flat.permute(1, 0, 3, 2, 5, 4)
    iarray_flat = iarray_flat.reshape(Mx*Nx, My*Ny, Mt*Nt)
    return iarray_flat

class poly_factorize_xyz():
    def __init__(self, n=5, order='xyz', x0=0, y0=0, z0=0,
                 cx=None, cy=None, cz=None):
        '''order lower than n
        '''
        self.n  = n
        self.order = 'xyz'
        self._cx = cx if cx else np.random.rand(self.n+1)
        self._cy = cy if cy else np.random.rand(self.n+1)
        self._cz = cz if cz else np.random.rand(self.n+1)

        self._x0 = x0
        self._y0 = y0
        self._z0 = z0

    @property
    def cx(self):
        return self._cx
    @cx.setter
    def cx(self, cx):
        self._cx = cx

    @property
    def cy(self):
        return self._cy
    @cy.setter
    def cy(self, cy):
        self._cy = _cy

    @property
    def cz(self):
        return self._cz
    @cz.setter
    def cz(self, cz):
        self._cz = cz

    def __call__(self, x, y, z):
        xval = x - self._x0
        yval = y - self._y0
        zval = z - self._z0
        if self.order == 'xyz':
            pass
        if self.order == 'yxz':
            raise NotImplementedError('Not implemented {}'\
                                      .format(self.order))
        if self.order == 'xzy':
            raise NotImplementedError('Not implemented {}'\
                                      .format(self.order))
        if self.order == 'yzx':
            raise NotImplementedError('Not implemented {}'\
                                      .format(self.order))
        if self.order == 'zxy':
            raise NotImplementedError('Not implemented {}'\
                                      .format(self.order))
        if self.order == 'zyx':
            xvalcopy = np.copy(xval)
            xval = zval
            zval = xvalcopy

        cx = self._cx
        cy = self._cy
        cz = self._cz
        yx = 0
        yy = 0
        yz = 0
        for i, ci in enumerate(cx):
            yx = yx + ci * xval**i
        for j, cj in enumerate(cy):
            yy = yy + cj * yval**j
        for k, ck in enumerate(cz):
            yz = yz + ck * zval**k

        return yx * yy * yz

    def __str__(self):
        sx = ''
        for i, ci in enumerate(self.cx):
            sx = sx + '{} * x**{}'.format(ci, i)
            if i != len(self.cx)-1:
                sx = sx + ' + '
        sy = ''
        for j, cj in enumerate(self.cy):
            sy = sy + '{} * y**{}'.format(cj, j)
            if j != len(self.cy)-1:
                sy = sy + ' + '
        sz = ''
        for k, ck in enumerate(self.cz):
            sz = sz + '{} * z**{}'.format(ck, k)
            if k != len(self.cz)-1:
                sz = sz + ' + '

        s = '({}) * ({}) * ({})'.format(sx, sy, sz)

        return s

def test_gpu_poly_poly(n=None, n2=None, qmodel=None,
                       Xs=[[0, 1, 11], [2, 3, 11], [3, 4, 11]],
                       **kwargs):
    cxi = kwargs.get('cxi', None)
    cyi = kwargs.get('cyi', None)
    czi = kwargs.get('czi', None)
    x0i = kwargs.get('x0i', 0)
    y0i = kwargs.get('y0i', 0)
    z0i = kwargs.get('z0i', 0)
    qistensor = kwargs.get('qistensor', False)

    np.random.seed(0)
    n = np.random.randint(0, 5) if n is None else n
    n2 = np.random.randint(2, 5) if n2 is None else n2

    # f = poly_factorize_xyz(5, cx=[0, 0, 0, 1],
    #                        cy=[0, 0, 0, 1], cz=[0, 0, 0, 1])
    # flinear = poly_factorize_xyz(1, cx=[0, 1], cy=[0, 1], cz=[0, 1])
    # flinear = poly_factorize_xyz(0, cx=[1], cy=[1], cz=[1])
    f = poly_factorize_xyz(n) if qmodel is None else qmodel
    flinear = poly_factorize_xyz(n2, cx=cxi, cy=cyi, cz=czi,
                                 x0=x0i, y0=y0i, z0=z0i)
    qeff = Qeff(xspace=Xs[0], yspace=Xs[1], tspace=Xs[2],
                model=f, flatten=False)
    qarray = qeff.create_qeff()

    # f is defined as f(x, y, z)
    # scipy evaluate functions by f(z, y, x), x0, x1, y0, y1, z0, z1
    # scipy.integrate.tqlquad(f)
    fxyz = lambda X, Y, Z : f(X, Y, Z) * flinear(X, Y, Z)
    yscipy = integrate.tplquad(fxyz, Xs[2][0], Xs[2][1],
                               Xs[1][0], Xs[1][1],
                               Xs[0][0], Xs[0][1], epsrel=1E-6)
    Xs = np.array(Xs)
    xval, yval, zval = np.meshgrid(np.linspace(Xs[0,0], Xs[0,1], Xs[0,2]),
                                   np.linspace(Xs[1,0], Xs[1,1], Xs[1,2]),
                                   np.linspace(Xs[2,0], Xs[2,1], Xs[2,2]),
                                   indexing='ij')

    # qeff after flattening
    qeff_flat = Qeff(xspace=Xs[0], yspace=Xs[1], tspace=Xs[2],
                     model=f, flatten=True)
    qarray_flat = qeff_flat.create_qeff()
    qeff_flat.func = flinear
    iarray_flat = qeff_flat.create_qeff_noweight()
    iarray_flat = correct_flat(iarray_flat,
                               Xs[0,2]-1, Xs[1,2]-1, Xs[2,2]-1,
                               4, 4, 4)

    jacob = (Xs[0,1]-Xs[0,0])/(Xs[0,2]-1) \
        * (Xs[1,1]-Xs[1,0])/(Xs[1,2]-1) * (Xs[2,1]-Xs[2,0])/(Xs[2,2]-1)
    norm_q_n = torch.sum(qarray_flat).item()
    if qistensor:
        qpoly = f(torch.tensor(xval), torch.tensor(yval),
                  torch.tensor(zval))
        ilinear = flinear(torch.tensor(xval),
                          torch.tensor(yval), torch.tensor(zval))
        norm_q_d = torch.sum(qpoly).item() * jacob
        simple_int = torch.sum(qpoly*ilinear).item() * jacob \
            *norm_q_n/norm_q_d
    else:
        qpoly = f(xval, yval, zval)
        ilinear = flinear(xval, yval, zval)
        norm_q_d = np.sum(qpoly).item() * jacob
        simple_int = np.sum(qpoly*ilinear) * jacob * norm_q_n/norm_q_d

    torch_int = torch.sum(qarray*ilinear).item()
    torch_crosscheck = torch.sum(qarray_flat * iarray_flat).item()
    msg = 'q order {}, i order {}, torch {}, simple {},'\
        ' scipy {}, '\
        ' scipy estimated error {}, torch crosscheck {}, '\
        'total diff {}, relative diff {}'\
        .format(n, n2, torch_int, simple_int,
                yscipy[0], yscipy[1], torch_crosscheck,
                (torch_int - yscipy[0]), (torch_int-yscipy[0])/yscipy[0])
    logger.debug(msg)
    return np.abs(torch_int-yscipy[0]), yscipy[0], yscipy[1], \
        norm_q_n/norm_q_d, np.abs(simple_int - yscipy[0]), \
        n, n2

# test_gpu_poly_linear = lambda n : test_gpu_poly_poly(n=n, n2=1)
test_gpu_poly_linear = partial(test_gpu_poly_poly, n2=1)
test_gpu_poly_xlinear = partial(test_gpu_poly_poly, n2=1, cxi=[0,1],
                               cyi=[1,0], czi=[1,0])

# def test_qmodel_xlinear(boundaries, x0, x1, y0, y1, t0, t1, sx, sy, st):
#     f = QModel.create_QModel((x0,y0,t0), (x1,y1,t1), (sx,sy,st))
#     output = test_gpu_poly_poly(qmodel=f, Xs=boundaries, n2=1,
#                                 cxi=[0, 1], x0=boundaries[0][0],
#                                 cyi=[0,0], czi=[0,0])
#     return output

def test_qmodel_xpoly(boundaries, x0, x1, y0, y1, t0, t1, sx, sy, st, n):
    f = QModel.create_QModel((x0,y0,t0), (x1,y1,t1), (sx,sy,st))
    output = test_gpu_poly_poly(qmodel=f, Xs=boundaries, n2=1,
                                cxi=[0,1], x0i=boundaries[0][0],
                                cyi=[1,0], czi=[1,0], qistensor=True)
    return output

test_qmodel_xlinear = partial(test_qmodel_xpoly, n=1)

def test_qmodel_poly_ntimes(k=None):
    output = { 'd' : [], 'y': [], 'dy': [],
               'csimple' : [], 'dsimple' : [],
               'n' : [], 'n2' : []}

    if k is None:
        k = np.random.randint(2, 7)

    boundaries = ((-5, 5, 11),(-5, 5, 11), (-5, 5, 11))
    xmin, xmax, nx = (-4, 4, 5)
    ymin, ymax, ny = (-4, 4, 5)
    tmin, tmax, nt = (-4, 4, 5)
    sigmas = list(range(1, 8, 2))

    X0 = np.linspace(xmin, xmax, nx)
    Y0 = np.linspace(ymin, ymax, nx)
    T0 = np.linspace(tmin, tmax, nt)

    X1 = X0[2:]
    Y1 = Y0[2:]
    T1 = T0[2:]
    X0 = X0[:-2]
    Y0 = Y0[:-2]
    T0 = T0[:-2]

    fdummy = lambda x, y, t : 1
    qeff = Qeff(xspace=boundaries[0], yspace=boundaries[1], tspace=boundaries[2],
         model = fdummy)

    for x0, x1 in zip(X0, X1):
        for y0, y1 in zip(Y0, Y1):
            if y0 < x0:
                continue
            for t0, t1 in zip(T0, T1):
                if t0 < y0:
                    continue
                for sx in sigmas:
                    for sy in sigmas:
                        if sy < sx:
                            continue
                        for st in sigmas:
                            if st < sx:
                                continue
                            logger.debug('processing {}, {}, {}...'.format((x0, y0, t0),(x1, y1, t1), (sx, sy, st)))
                            result = test_qmodel_xpoly(
                                boundaries,
                                x0, x1, y0, y1, t0, t1, sx, sy, st,
                                n = k
                            )

                            d, y, dy, csimple, dsimple, \
                                n, n2 = result
                            output['d'].append(d)
                            output['y'].append(y)
                            output['dy'].append(dy)
                            output['csimple'].append(csimple)
                            output['dsimple'].append(dsimple)
                            output['n'].append(n)
                            output['n2'].append(n2)
                            msg = 'abs diff {}, rel diff {}, rel diff simple {}, '\
                                'csimple {}'.\
                                format(output['d'][-1],
                                       output['d'][-1]/output['y'][-1],
                                   output['dsimple'][-1]/output['y'][-1],
                                   output['csimple'][-1])
                            logger.debug(msg)
    for k, v in output.items():
        output[k] = np.array(v)
    print('abs diff min',
          np.min(output['d']),
          'abs diff max', np.max(output['d']),
          'rel diff min', np.min(output['d']/output['y']),
          'rel diff max', np.max(output['d']/output['y']),
          'rel diff simple min', np.min(output['dsimple']/output['y']),
          'rel diff simple max', np.max(output['dsimple']/output['y']),
          'csimple', np.min(output['csimple']), np.max(output['csimple'])
          )

test_qmodel_xlinear_ntimes = partial(test_qmodel_poly_ntimes, k=1)

def test_gpu_poly_ntimes(n, f):
    output = { 'd' : [], 'y': [], 'dy': [],
               'csimple' : [], 'dsimple' : [],
               'n' : [], 'n2' : []}
    np.random.seed(0)
    k = np.random.randint(0, 8, size=n)
    k2 = np.random.randint(2, 8, size=n)
    if f == test_gpu_poly_linear or f == test_gpu_poly_xlinear:
        k2 = np.ones(k2.shape, dtype=int)
    for i in range(n):
        if f == test_gpu_poly_linear or f == test_gpu_poly_xlinear:
            d, y, dy, csimple, dsimple, \
            n, n2 = f(k[i])
        else:
            d, y, dy, csimple, dsimple, \
                n, n2 = f(k[i], k2[i])
        output['d'].append(d)
        output['y'].append(y)
        output['dy'].append(dy)
        output['csimple'].append(csimple)
        output['dsimple'].append(dsimple)
        output['n'].append(n)
        output['n2'].append(n2)
    for k, v in output.items():
        output[k] = np.array(v)
    print('abs diff min',
          np.min(output['d']),
          'abs diff max', np.max(output['d']),
          'rel diff min', np.min(output['d']/output['y']),
          'rel diff max', np.max(output['d']/output['y']),
          'rel diff simple min', np.min(output['dsimple']/output['y']),
          'rel diff simple max', np.max(output['dsimple']/output['y']),
          'csimple', np.min(output['csimple']), np.max(output['csimple'])
          )

if __name__ == '__main__':
    logging.basicConfig(filename='run_toy_gpu2.log', filemode='w',
                        level=logging.WARNING)
    logger.setLevel(logging.DEBUG)

    logger.debug('Running test_gpu_poly_linear')
    print('Running test_gpu_poly_linear')
    test_gpu_poly_ntimes(100, test_gpu_poly_linear)
    logger.debug('Running test_gpu_poly_xlinear')
    print('Running test_gpu_poly_xlinear')
    test_gpu_poly_ntimes(100, test_gpu_poly_xlinear)
    logger.debug('Running test_gpu_poly_poly')
    print('Running test_gpu_poly_poly')
    test_gpu_poly_ntimes(100, test_gpu_poly_poly)

    logger.debug('Running test_qmodel_xlinear_ntimes')
    test_qmodel_xlinear_ntimes()
    # test_qmodel_poly_ntimes(1)
    # logger.setLevel(logging.DEBUG)

    exit(0)
