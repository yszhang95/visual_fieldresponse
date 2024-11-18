import numpy as np
import scipy as sp

import logging

logger = logging.getLogger(__name__)

class Func():
    def __init__(self, f=None):
        if isinstance(f, str):
            raise NotImplementedError("TBD, str to predefined function.")
        if callable(f):
            raise NotImplementedError("TBD, support for input function.")
        if f is None:
            raise NotImplementedError("TBD, add a default test function.")
        else:
            raise NotImplemented(f"Not supported for type {type(f)}")
        self._f = None
    @property
    def f(self):
        return f

class ii():
    '''induced current'''
    pass

class Qdist(Func):
    '''distribution of Q
    '''
    def __init__(self):
        # __super__.init(self)
        pass

class Qeff():
    '''array for effective q
    size
    grid size
    '''
    def __init__(self, xspace, yspace, tspace, meshgrid=True,
                 method='gauss_legendre_2'):
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
        # Gauss-Legendre n points
        if method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in method[0:len('gauss_legendre')]:
            self.__np = int(method[len('gauss_legendre')+1:])
            self.__method = method
        else:
            msg = (
                f"Method {method} not supported. "
                "Available options are cube_corner, gauss_legendre_n, "
                "where n means n-point Gauss-Legendre Quadrature. "
                "For instance, gauss_legendre_2 means two-point "
                "Gauss-Legendre Quadrature."
            )
            raise NotImplementedError(msg)

        self.__coord_1d = {k : np.linspace(v[0], v[1], v[2])
                           for k, v in self.__space.items()}
        self.__grid_1d, self.__w_grid_1d, w_grid_1d_unit = self._create_grid1d()

        self.__w_grid_unit_block = Qeff._create_weight_block(w_grid_1d_unit)

        if self.__meshgrid:
            self.__coordinate = np.meshgrid(self.__coord_1d['x'],
                                            self.__coord_1d['y'],
                                            self.__coord_1d['t'], indexing='ij')

            self.__gridpoints = np.meshgrid(self.__grid_1d['x'],
                                            self.__grid_1d['y'],
                                            self.__grid_1d['t'], indexing='ij')
            # w3d[i,j,k]=wx[i]*wy[j]*wt[k]
            self.__w_grid_3d = np.multiply.outer(
                np.multiply.outer(
                    self.__w_grid_1d['x'], self.__w_grid_1d['y']
                ), self.__w_grid_1d['t']
            )


    @property
    def method(self):
        return self.__method

    @staticmethod
    def _create_weight_block(w1d):
        '''create a weight block'''
        nx = len(w1d['x'])
        ny = len(w1d['y'])
        nt = len(w1d['t'])
        w3d = np.zeros([nx, ny, nt])
        for i in range(nx):
            for j in range(ny):
                for k in range(nt):
                    w3d[i,j,k] = w1d['x'][i] * w1d['y'][j] * w1d['t'][k]
        return w3d

    def _create_grid1d(self):
        '''create grid and corresponding weights in 1d.
        Note:
        1. A uniform spacing of cubes is assumed.
        2. (b-a)/2 of each interval is included in the weight
        '''
        grid_1d = {}
        w_grid_1d = {}
        w_grid_1d_unit = {}
        if self.__method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in self.__method:
            n = self.__np
            roots, weights = sp.special.roots_legendre(n)
            for k in ['x', 'y', 't']:
                corners = np.linspace(self.__space[k][0],
                                self.__space[k][1], self.__space[k][2])
                # to be optimized
                grid_1d[k] = np.concatenate(list(
                    (b-a)/2 * roots + (b+a)/2
                    for (a, b) in zip(corners[:-1], corners[1:])
                ))
                # assume equal spacing
                step = np.abs(corners[1] - corners[0])
                w_grid_1d[k] = np.tile(weights*step/2., self.__space[k][2])
                w_grid_1d_unit[k] = w_grid_1d[k][0:self.__np]
        else:
            pass
        return grid_1d, w_grid_1d, w_grid_1d_unit

    def _eval_index_coord_no_check(self, i, j, k):
        '''evaluate the coordinate of corners at x[i], y[j], t[k]
        '''
        if self.__meshgrid:
            return (
                self.__coordinate[0][i,j,k],
                self.__coordinate[1][i,j,k],
                self.__coordinate[2][i,j,k]
            )
        else:
            return tuple(
                # self.__space[key][0] + idx * self.__gridspacing[key]
                self.__coord_1d[key][idx]
                for key, idx in zip(['x', 'y', 't'], [i,j,k])
            )

    def _eval_cube_coord(self, i, j, k):
        '''evaluate two corners of a cube, (x0,y0,t0)<->(x1,y1,t1)'''
        try:
            return (
                self._eval_index_coord_no_check(i, j, k),
                self._eval_index_coord_no_check(i+1, j+1, k+1)
            )
        except IndexError:
            raise IndexError(f'i,j,k<={i},{j},{k} must be smaller than'
                             f'{self.__space["x"][2]-1}, '
                             f'{self.__space["y"][2]-1}, '
                             f'{self.__space["y"][2]-1}.')

    def _eval_grid_block(self, i, j, k):
        '''evaluate block at (ith,jth,kth) cube of grid'''
        if self.__meshgrid:
            grid_points = np.array(
                [block[i*self.__np:(i+1)*self.__np,
                       j*self.__np:(j+1)*self.__np,
                       k*self.__np:(k+1)*self.__np] \
                 for block in self.__gridpoints])
            grid_points = np.transpose(grid_points, (1, 2, 3, 0))
        else:
            grid_points = np.array(tuple(
                (x, y, t) for x in \
                self.__grid_1d['x'][i*self.__np:(i+1)*self.__np] \
                for y in self.__grid_1d['y'][j*self.__np:(j+1)*self.__np] \
                for t in self.__grid_1d['t'][k*self.__np:(k+1)*self.__np] \
            )).reshape(self.__np, self.__np, self.__np, 3)
        return grid_points

    def _test_eval_coord(self,i,j,k):
        '''test both _eval_index_coord_no_check and _eval_cube_coord'''
        self._logger.debug(f'testing i={i}, j={j}, k={k}...')
        for key in ['x', 'y', 't']:
            self._logger.debug(
                f'charge space along {key}, {self.__space[key]}')
        coord = self._eval_index_coord_no_check(i,j,k)
        try:
            cube_coord = self._eval_cube_coord(i,j,k)
        except IndexError as e:
            cube_coord = None
            print(e)
        self._logger.debug(f'eval coord at index {i},{j},{k} -> {coord}.')
        self._logger.debug(f'eval coord at cube {i},{j},{k} -> '
                           f'{cube_coord}.')

    def _test_create_grid1d(self):
        grid_1d, w_grid_1d, _ = self._create_grid1d()
        for k in ['x', 'y', 't']:
            for i in range(min(3, len(grid_1d[k])-1)):
                cube = np.linspace(self.__space[k][0], self.__space[k][1],
                            self.__space[k][2])
                msg = 'grid_point in cell <-({}, {}), are, {}. '\
                    'Their weights are {}. '\
                    'The original Gauss-Legendre coef/weights are {}'\
                    .format(cube[i], cube[i+1],
                            grid_1d[k][i*self.__np:(i+1)*self.__np],
                            w_grid_1d[k][i:i+self.__np],
                            sp.special.roots_legendre(self.__np)
                )
                self._logger.debug(msg)

    def _test_grid3d(self, i=None, j=None, k=None):
        '''
        test grid points by _eval_grid_block(i,j,k)
        test weight grid by _create_weight_block(w1d)
        '''
        def printout(i,j,k):
            grid_points = self._eval_grid_block(i,j,k)
            msg = 'grid_point at cube {}<->({}), are {}'\
                .format((i,j,k), self._eval_cube_coord(i,j,k),
                        grid_points)
            self._logger.debug(msg)
            if self.__meshgrid:
                w3d_unit = self.__w_grid_3d[i*self.__np:(i+1)*self.__np,
                                            j*self.__np:(j+1)*self.__np,
                                            k*self.__np:(k+1)*self.__np]
                msg = 'weights of at cube{}<->({}), are {},'\
                    ' diffrences w.r.t for loop are {}'\
                    .format((i,j,k), self._eval_cube_coord(i,j,k),
                            w3d_unit, w3d_unit-self.__w_grid_unit_block)
                self._logger.debug(msg)

        if i is None or j is None or k is None:
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0 ,3):
                        printout(i,j,k)
        else:
            printout(i,j,k)

    def _test_qgaus(self):
        pass

class integrator():
    '''integrator'''
    pass

# validate; integrating qeff * ii
# what is the range?
# most strict way is to integrate over all space of qeff

if __name__ == '__main__':
    print("Running test")
    logging.basicConfig(level=logging.DEBUG)

    # validated
    # qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
    #             meshgrid=True, method='gauss_legendre_3')
    qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
                meshgrid=True, method='gauss_legendre_2')
    # qeff = Qeff(xspace=(0, 1, 11), yspace=(2, 3, 11), tspace=(3, 4, 11),
    #            meshgrid=False)
    # validated
    # qeff._test_eval_coord(2, 3, 4)
    # validated
    # qeff._test_eval_coord(10, 10, 10)
    # validated
    # qeff._test_create_grid1d()
    # qeff._test_grid3d(0, 1, 2)
