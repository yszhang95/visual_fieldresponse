import torch

import numpy as np
import scipy as sp

import json

class UniversalGrid:
    """
    A class representing a universal grid in N dimensions. The grid is defined by its
    origin and grid spacing along each dimension. 

    This class supports operations such as converting between physical coordinates and grid
    indices, and it accommodates an optional origin offset for shifting the reference point.

    TBD: alignment for origin with some reference, like edges of pixels

    Attributes:
        _grid_spacing (torch.Tensor): Grid spacing along each dimension.
        _origin (torch.Tensor): Physical coordinates of the grid's origin.
    """

    def __init__(self, grid_spacing=None, origin=None):
        """
        Initializes the grid using separate arrays for each dimension.

        Args:
            grid_spacing (tuple): Spacing between grid points in each dimension.
            origin (tuple, optional): Physical coordinates of the grid's origin. If None,
                the origin defaults to `min_limits`. If provided, the origin must lie within
                the grid bounds.

        Raises:
            NotImplementedError: If the provided origin is outside the minimum limits.
        """
        
        self._grid_spacing = torch.tensor(grid_spacing, dtype=torch.float32, requires_grad=False) if grid_spacing else None
        self._origin = torch.tensor(origin, dtype=torch.float32, requires_grad=False) if origin else None
    @property
    def origin(self):
        return self._origin

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @staticmethod
    def compute_coordinate(idxs, spacing, origin):
        '''
        idxs : (N, vdim); index of grid point
        spacing : (vdim, )
        origin : (vdim,)
        return origin + spacing * idx
        '''
        if not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.int64, requires_grad=False)
        if len(idxs.shape) == 1:
            idxs = idxs.unsqueeze(0)
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32, requires_grad=False)
        if not isinstance(spacing, torch.Tensor):
            spacing = torch.tensor(spacing, dtype=torch.float32, requires_grad=False)
        return origin.unsqueeze(0) + idxs * spacing.unsqueeze(0)
        
    @staticmethod
    def compute_index(coords, origin, spacing):
        '''
        spacing : (vdim, )
        coords : (N, vdim)
        origin : (vdim,)

        return (coords - origin)/spacing, index of grid point
        '''
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32, requires_grad=False)
        if len(coords.shape) == 1:
            coords = coords.unsqueeze(0)
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32, requires_grad=False)
        if not isinstance(spacing, torch.Tensor):
            spacing = torch.tensor(spacing, dtype=torch.float32, requires_grad=False)
        idxs = (coords - origin.unsqueeze(0)) / spacing.unsqueeze(0)
        idxs = idxs.floor().to(torch.int64)
        return idxs

    def get_coordinate(self, idxs):
        '''
        wraper
        '''
        return UniversalGrid.compute_coordinate(idxs, self._grid_spacing, self._origin)

    def get_index(self, coords):
        """
        Converts physical coordinates to grid indices, considering the grid's origin offset.

        Args:
            coords (torch.Tensor): Physical coordinates (M, vdim).
            offset (torch.Tensor, optional): Offset to apply to indices. Defaults to the
                grid's origin offset.

        Returns:
            torch.Tensor: Grid indices (M, vdim).
        """
        return UniversalGrid.compute_index(coords, self._origin, self._grid_spacing)

    def from_grid(self, grid_spacing, origin=None):
        '''
        grid : tuple of torch.Tensor
        origin: new origin, default : None --> minimum of grid
        '''
        if not isinstance(grid_spacing, torch.Tensor):
            grid_spacing = torch.tensor(grid_spacing, dtype=torch.float32, requires_grad=False)
        self._grid_spacing = grid_spacing
        if not isinstance(origin, torch.Tensor):
            origin = torch.tensor(origin, dtype=torch.float32, requires_grad=False)
        self._origin = origin
        return self

    def coarse_grain(self, multiples):
        '''
        multiples : (vdim, )
        '''
        if not isinstance(multiples, torch.Tensor):
            multiples = torch.tensor(multiples, dtype=torch.float32, requires_grad=False)
        if len(self._grid_spacing) != len(multiples):
            raise ValueError('In compatible shape of multiples, and grid spacing, {} incompatible to {}'.format(multiples.shape, self._grid_spacing.shape))
        return UniversalGrid().from_grid(self._grid_spacing * multiples, self._origin)



class LocalGrid:
    """
    A class for building a local grid to compute offsets to origin in terms of index,
    and shapes in unit of number of grid spacing, for a series of steps.
    """

    def __init__(self, origin, grid_spacing, n_sigma):
        """
        Initialize the grid.

        Args:
            origin (vdim,) (float): Origin for grid in each dimension.
            grid_spacing (vdim,) (float): Spacing for grid in each dimension.
            n_sigma (vdim,), (int): Number of sigmas to include in bounds.
        """
        self._grid_spacing = grid_spacing.copy(requires_grad=False) if isinstance(grid_spacing, torch.Tensor) else torch.tensor(grid_spacing, dtype=torch.float32, requires_grad=False)
        self._origin = origin.copy(requires_grad=False) if isinstance(origin, torch.Tensor) else torch.tensor(origin, dtype=torch.float32, requires_grad=False)
        self._n_sigma = n_sigma.copy(requires_grad=False) if isinstance(n_sigma, torch.Tensor) else torch.tensor(n_sigma, dtype=torch.float32, requires_grad=False)

    @staticmethod
    def compute_bounds_X0X1(X0X1, Sigma, n_sigma):
        '''
        X0X1 : (N, vdim, 2)
        Sigma : (N, vdim)
        n_sigma: (vdim, )
        return : (N, vdim, 2)
        '''
        if not isinstance(n_sigma, torch.Tensor):
            n_sigma = torch.tensor(n_sigma, dtype=torch.float32, requires_grad=False)
        offset = (n_sigma.unsqueeze(0) * Sigma) # (N, vdim)
        min_limits = torch.min(X0X1, dim=2).values - offset # torch.min(shape(N,vdim,2)) --> shape(N, vdim)
        max_limits = torch.max(X0X1, dim=2).values + offset
        return torch.stack([min_limits, max_limits], dim=2)

    @staticmethod
    def stack_X0X1(X0, X1):
        '''
        X0: (N, vdim)
        X1: (N, vdim)
        return (N, vdim, 2)
        '''
        return torch.stack((X0, X1), dim=2)

    @staticmethod
    def compute_bounds_X0_X1(X0, X1, Sigma, n_sigma):
        '''
        X0: (N, vdim)
        X1: (N, vdim)
        Sigma: (N, vdim)
        n_sigma: (vdim, )
        return (N, vdim, 2) float
        '''
        combined = LocalGrid.stack_X0X1(X0, X1)
        bounds = LocalGrid.compute_bounds_X0X1(combined, Sigma, n_sigma)
        return bounds

    @staticmethod
    def reduce_to_universal(shape):
        """
        Compute a universal shape across all steps using reductions.

        Args:
            shapes (torch.Tensor): Shapes of the charge boxes (N, vdim).

        Returns:
            Universal shape. (vdim, )
        """
        universal_max = torch.max(shape, dim=0).values

        return universal_max
    
    @staticmethod
    def compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing, compare_key='index'):
        '''
        offset : (N, vdim)
        shape : (vdim,)
        '''
        extremes = LocalGrid.compute_bounds_X0_X1(X0, X1, Sigma, n_sigma) # (N, vdim, 2)
        min_limit = extremes[:, :, 0] # (N, vdim)
        max_limit = extremes[:, :, 1] # (N, vdim)
        if compare_key == 'index':
            min_limit = UniversalGrid.compute_index(min_limit, origin, grid_spacing)
            max_limit = UniversalGrid.compute_index(max_limit, origin, grid_spacing)
            offset = min_limit
            shape = max_limit - min_limit + 1
            shape = LocalGrid.reduce_to_universal(shape)
        elif compare_key == 'coordinate':
            raise NotImplementedError('Not support comparation by coordinate')
            '''
            The index computation cast the coordinate to lower bound.
            I am not sure how to reconcile cases below:
            1) Minimum close to lower bound and maximum close to upper bound.
            2) Minimum close to upper bound and maximum close to lower bound.
            The shape given by the two are off by one.
            '''
        else:
            raise NotImplementedError('Only support comparation by index and coordinate')
        return offset, shape

    
    def compute_bounds(self, X0, X1, Sigma):
        """
        Compute the adjusted bounds for the grid.

        Args:
            X0 (torch.Tensor): Starting points of the steps (N, vdim).
            X1 (torch.Tensor): Ending points of the steps (N, vdim).
            Sigmas (torch.Tensor): Gaussian diffusion widths (N, vdim).

        Returns:
            tuple: Adjusted min and max bounds for the grid (min_limits, max_limits).
        """
        # Stack X0 and X1 to find global min and max
        bounds = LocalGrid.compute_bounds_X0_X1(X0, X1, Sigma, self._n_sigma)
        return bounds

    def compute_index(self, coords):
        """
        Compute grid indices directly from physical coordinates.

        Args:
            coords (torch.Tensor): Physical coordinates (N, vdim).
            grid_min (torch.Tensor): Minimum bounds of the grid, (vdim, ).

        Returns:
            torch.Tensor: Grid indices (N, vdim).
        """
        return UniversalGrid.compute_index(coords, self._origin, self._grid_spacing)

    def get_charge_box(self, X0, X1, Sigma, compare_key='index'):
        """
        Compute the offset and shape of the charge box for a series of steps.

        Args:
            X0 (torch.Tensor): Starting points of the steps (N, vdim).
            X1 (torch.Tensor): Ending points of the steps (N, vdim).
            Sigmas (torch.Tensor): Gaussian diffusion widths (N, vdim).

        Returns:
            dict: Offsets and shapes of the charge boxes.
        """
        return LocalGrid.compute_charge_box(X0, X1, Sigma, self._n_sigma, self._origin, self._grid_spacing, compare_key)

class QModel():
    def __new__(cls, *args, **kwargs):
        raise TypeError("This class cannot be instantiated.")
    @staticmethod
    def GaussConvLine3D(Q, X0, X1, Sigma, x, y, z):
        '''
        Q (N,)
        X0 (N, 3)
        X1 (N, 3)
        Sigma (N, 3)
        x (N, other shape)
        y (N, other shape)
        z (N, other shape)
        '''
        sqrt2 = np.sqrt(2)
        if not isinstance(Q, torch.Tensor):
            Q = torch.tensor(Q, dtype=torch.float32, requires_grad=False)
        if not isinstance(X0, torch.Tensor):
            X0 = torch.tensor(X0, dtype=torch.float32, requires_grad=False)
        if not isinstance(X1, torch.Tensor):
            X1 = torch.tensor(X1, dtype=torch.float32, requires_grad=False)
        if not isinstance(Sigma, torch.Tensor):
            Sigma = torch.tensor(Sigma, dtype=torch.float32, requires_grad=False)
        if len(X0.shape) != 2 or X0.shape[1] != 3:
            raise ValueError(f'In compatible shape of X0. Converting from {X0.shape} to (N, 3) before processing')
        if len(X1.shape) != 2 or X1.shape[1] != 3:
            raise ValueError(f'In compatible shape of X1. Converting from {X1.shape} to (N, 3) before processing')
        if len(Sigma.shape) != 2 or Sigma.shape[1] != 3:
            raise ValueError(f'In compatible shape of Sigma. Converting from {Sigma.shape} to (N, 3) before processing')

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=False)
        if len(x) != len(X0):
            raise ValueError('Incompatible shapes between x and X0,X1,Sigma')
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, requires_grad=False)
        if len(y) != len(X0):
            raise ValueError('Incompatible shapes between x and X0,X1,Sigma')
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, requires_grad=False)
        if len(z) != len(X0):
            raise ValueError('Incompatible shapes between x and X0,X1,Sigma')

        x0, y0, z0 = X0[:,0], X0[:,1], X0[:,2] # (N,) (N,), (N,)
        x1, y1, z1 = X1[:,0], X1[:,1], X1[:,2] # (N,) (N,), (N,)
        sx, sy, sz = Sigma[:,0], Sigma[:,1], Sigma[:,2] # (N,) (N,), (N,)

        # brodcasting following

        dx01 = x0 - x1 # (N,)
        dy01 = y0 - y1 # (N,)
        dz01 = z0 - z1 # (N,)
        sxsy2 = (sx*sy)**2 # (N,)
        sxsz2 = (sx*sz)**2 # (N,)
        sysz2 = (sy*sz)**2 # (N,)

        sx2 = sx**2 # (N,)
        sy2 = sy**2 # (N,)
        sz2 = sz**2 # (N,)
        
        deltaSquare = (
            sysz2 * dx01**2
            + sxsy2 * dz01**2 + sxsz2 * dy01**2
        ) # (N,)

        deltaSquareSqrt = torch.sqrt(deltaSquare) # (N,)

        deltaSquareSqrt4pi = deltaSquareSqrt * 4 * np.pi # (N,)
        erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz # (N,)
        '''
        argA1 = (
            sysz2 * (x - x0)*dx01 + # (N,Nx,1,1)
            sxsy2 * (z - z0)*dz01 + # (N,1,1,Nz)
            sxsz2 * (y - y0)*dy01   # (N,1,Ny,1)              
        ) # (N, Nx, Ny, Nz)

        argA2 = (
            sysz2 * (x - x1) * dx01 + # (N,Nx,1,1)
            sxsy2 * (z - z1) * dz01 + # (N,1,1,Nz)
            sxsz2 * (y - y1) * dy01 # (N,1,Ny,1) 
        ) # (N, Nx, Ny, Nz)

        argB = (
            sy2 * torch.pow(x * dz01 + (z1*x0 - z0*x1) - z * dx01, 2) + # (N, Nx, 1, Nz)
            sx2 * torch.pow(y * dz01 + (z1*y0 - z0*y1) - z * dy01, 2) + # (N, 1, Ny, Nz)
            sz2 * torch.pow(y * dx01 + (x1*y0 - x0*y1) - x * dy01, 2) # (N, Nx, Ny, 1)
        ) # (N, Nx, Ny, Nz)
        
        charge = (
            -Q* torch.exp(-0.5*argB/deltaSquare)/deltaSquareSqrt4pi * (
                torch.erf(argA1/erfArgDenominator) -
                torch.erf(argA2/erfArgDenominator)
            )
        )
        '''
        charge = (
            -Q * torch.exp(-0.5 * (
                sy2 * torch.pow(x * dz01 + (z1*x0 - z0*x1) - z * dx01, 2) + 
                sx2 * torch.pow(y * dz01 + (z1*y0 - z0*y1) - z * dy01, 2) + 
                sz2 * torch.pow(y * dx01 + (x1*y0 - x0*y1) - x * dy01, 2) 
                                )/deltaSquare)/deltaSquareSqrt4pi * (
                torch.erf((
                    sysz2 * (x - x0)*dx01 + # (N,Nx,1,1)
                    sxsy2 * (z - z0)*dz01 + # (N,1,1,Nz)
                    sxsz2 * (y - y0)*dy01   # (N,1,Ny,1)              
                        )/erfArgDenominator) - 
                torch.erf((
                    sysz2 * (x - x1) * dx01 + 
                    sxsy2 * (z - z1) * dz01 + 
                    sxsz2 * (y - y1) * dy01  
                        )/erfArgDenominator)
                )
            )

        return charge


class QEff3D():
    '''QEff'''
    def __init__(self, origin, grid_spacing, offset, shape,
                 method='gauss_legendre_4_4_4'
                 ):
        '''xspace, yspcae, tspace are tuples of three numbers
        defining the range and number of grid points along one dimension
        of the charge space.
        Note: The end point is always included.
        Input (1, 2, 3) will yield ([1, 1.5, 2].
        '''
        # bad as it needs to be (N, other shapes)
        # TBD
        # self._grid = tuple(
        #     torch.arange(o+off*step, o+(off+shp)*step+0.1*step,step) for o, step, off, shp, in zip(origin, grid_spacing, offset, shape)
        # )

        self._box_shape = shape
        self._box_offset = offset if isinstance(offset, torch.Tensor) else torch.tensor(offset, dtype=torch.int64, requires_grad=False)
        self._origin = origin
        self._grid_spacing = grid_spacing

        self._method, self._npoints = QEff3D.parse_method(method)

    @property
    def origin(self):
        return self._origin
    @property
    def grid_spacing(self):
        return self._grid_spacing
    @property
    def box_shape(self):
        return self._box_shape
    @property
    def box_offset(self):
        return self._box_offset

    @staticmethod
    def parse_method(method):
        if method == 'cube_corner':
            raise NotImplementedError("cube_corner not implemented.")
        elif 'gauss_legendre' in method[0:len('gauss_legendre')]:
            npoints = method[len('gauss_legendre')+1:].split('_')
            npoints = tuple(int(p) for p in npoints)
            return 'gauss_legendre', npoints
        else:
            msg = (
                f"Method {method} not supported. "
                "Available options are cube_corner, gauss_legendre_n_n_n, "
                "where n means n-point Gauss-Legendre Quadrature. "
                "For instance, gauss_legendre_2 means two-point "
                "Gauss-Legendre Quadrature."
            )
            raise NotImplementedError(msg)

    @staticmethod
    def create_w1d_GL(npt, spacing):
        '''
        npt : integer
        spacing : float
        return a tensor of weights of n-point GL quadrature after correcting for length of intervals
        '''
        _, weights = sp.special.roots_legendre(npt)
        w1d = torch.tensor(weights, dtype=torch.float32, requires_grad=False) * spacing/2
        return w1d

    @staticmethod
    def create_w1ds(method, npoints, grid_spacing):
        '''
        method : str
        npoints : (3, ), integers
        grid_spacing : (3, ), float
        return (3, ) list with each element is a tensor of weights of n-point GL quadrature after correcting for length of intervals
        '''
        if method != 'gauss_legendre':
            raise NotImplementedError('Not implemented method but gauss legendre quadrature')
        w1ds = [None, None, None]
        for ipt, npt in enumerate(npoints):
            w1ds[ipt] = QEff3D.create_w1d_GL(npt, grid_spacing[ipt])
        return w1ds

    @staticmethod
    def create_weight_block(w1ds):
        '''create a weight block in 3D'''
        w3d = w1ds[0][:, None, None] * w1ds[1][None, :, None] * w1ds[2][None, None, :]
        return w3d

    @staticmethod
    def create_u1d_GL(npt):
        '''
        npt : integer
        return: a tensor of coefficients for interpolations at roots of npt-order GL polynomials
        '''
        roots, _ = sp.special.roots_legendre(npt)
        roots = torch.tensor(roots, dtype=torch.float32, requires_grad=False)
        u = (roots+1)/2
        u1d = torch.empty([npt, 2], dtype=torch.float32, requires_grad=False)
        u1d[:, 0] = 1-u
        u1d[:, 1] = u
        return u1d

    @staticmethod
    def create_u1ds(method, npoints):
        '''
        method : str
        npoints : (3, ) integers
        return (3, ) list with each elment is a tensor of coefficients for interpolations
        '''
        u1ds = []
        for ipt, npt in enumerate(npoints):
            u1ds.append(QEff3D.create_u1d_GL(npt))
        return u1ds

    @staticmethod
    def create_u_block(u1ds):
        '''create a weight block for u in 3D
        Requirements: u1d with a shape of (npt, 2) where npt means n-point GL quad rule.
        '''
        for i in range(3):
            if u1ds[i].shape[1] != 2:
                raise ValueError('u1d must have a shape of (npt, 2)')
        u3d = (
            u1ds[0][:, None, None, :, None, None] *
                u1ds[1][None, :, None, None, :, None] *
                u1ds[2][None, None, :, None, None, :]
        )
        return u3d

    @staticmethod
    def create_wu_block(w3d, u3d):
        '''
        w3d (L,M,N)
        u3d (L,M,N,2,2,2)
        return w3d[i,j,k] * u3d{i,j,k,:,:,:]
        '''
        return w3d[:,:,:,None,None,None] * u3d

    @staticmethod
    def create_sampling_1d_GL(origin_1d, grid_spacing_1d, offset_1d, shp_1d, npt):
        '''
        origin_1d : float
        grid_spacing_1d : float
        offset_1d : (Nsteps, ) tensor of integer
        shp_1d : integer, I/J/K where I/J/K is the number of grid points along x/y/z
        npt : npoints GL quad rule
        return: samplings (Nsteps, L/M/N, I/J/K-1).
        '''
        shp_idx_1d = torch.arange(shp_1d+1, dtype=torch.float32, requires_grad=False) # (I/J/K, )
        idx_2d = offset_1d[:,None] + shp_idx_1d[None, :]
        corners_1d = origin_1d + idx_2d * grid_spacing_1d # (Nsteps, I/J/K)
        roots, _ = sp.special.roots_legendre(npt) # (L/M/N, )
        roots = torch.tensor(roots, dtype=torch.float32, requires_grad=False)
        half_delta = (corners_1d[:, 1:] - corners_1d[:, :-1])/2. # (Nsteps, I/J/K-1)
        avg = (corners_1d[:,1:] + corners_1d[:,:-1])/2. # (Nsteps, I/J/K)
        sampling_1d = half_delta[:, None, :] * roots[None, :, None] + avg[:, None, :] # (Nsteps, L/M/N, I/J/K-1)
        return sampling_1d

    @staticmethod
    def create_sampling_1ds(method, npoints, origin, grid_spacing, offset, shape):
        '''
        method : str
        origin : tuple of three floats
        grid_spaing : tuple of three floats
        offset : (Nsteps, 3) for three dimensions
        shape : (3, ) for three dimensions
        npoints : (3, ) for GL quad rule
        return: list of sampling points with each element is a tensor with a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        sampling_1ds = []
        if method != 'gauss_legendre':
            raise NotImplementedError('Not implemented method but gauss legendre quadrature')
        for i in range(3):
            sampling_1ds.append(QEff3D.create_sampling_1d_GL(origin[i], grid_spacing[i],
                                                             offset[:,i], shape[i], npoints[i]))
        return sampling_1ds

    @staticmethod
    def eval_line_conv_gaus(Q, X0, X1, Sigma, x, y, z):
        '''
        Q (Nsteps, ),
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        charge = QModel.GausConvLine(
            Q, X0, X1, Sigma,
            x[:, :, None, None, :, None, None], # (Nsteps, L, 1, 1, I-1, 1, 1)
                        y[:, None, :, None, None, :, None], # (Nsteps, 1, M, 1, 1, J-1, 1)
                        z[:, None, None, :, None, None, :]) # (Nsteps, 1, 1, N, 1, 1, K-1)
        return charge
        
    @staticmethod
    def eval_qmodel(Q, X0, X1, Sigma, x, y, z, **kwargs):
        '''
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        qmodel = kwargs.get('qmodel', None)
        if qmodel is None:
            qmodel = QEff3D.eval_line_conv_gaus
        charge = qmodel(Q, X0, X1, Sigma, x[:, :, None, None, :, None, None], # (Nsteps, L, 1, 1, I-1, 1, 1)
                        y[:, None, :, None, None, :, None], # (Nsteps, 1, M, 1, 1, J-1, 1)
                        z[:, None, None, :, None, None, :]) # (Nsteps, 1, 1, N, 1, 1, K-1)
        return charge

    @staticmethod
    def eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing, method, npoints, **kwargs):
        '''
        '''
        u1ds = QEff3D.create_u1ds(method, npoints)
        w1ds = QEff3D.create_w1ds(method, npoints, grid_spacing)
        ublock = QEff3D.create_u_block(u1ds)
        wblock = QEff3D.create_weight_block(w1ds)
        
        x, y, z = QEff3D.create_sampling_1ds(method, npoints, origin, grid_spacing, offset, shape)

        lmn = tuple(wblock.shape)
        lmn_prod = torch.prod(torch.tensor(lmn, dtype=torch.int64), 0)
        rst = ublock.shape[3:]

        kernel = QEff3D.create_wu_block(wblock, ublock)
        kernel = torch.flip(kernel, [3, 4, 5]) # it does not matter we flip at first or we multiply w and u at first
        kernel = kernel.view(lmn_prod, 1, rst[0], rst[1], rst[2]) # out_channel, in_channel/groups, R, S, T

        charge = QEff3D.eval_qmodel(Q, X0, X1, Sigma, x, y, z, **kwargs)

        qeff = charge.view(len(charge), lmn_prod, shape[0], shape[1], shape[2]) # batch, channel, D1, D2, D3
        qeff = torch.nn.functional.pad(qeff, pad=(rst[2]-1, rst[2]-1, rst[1]-1, rst[1]-1,
                                                rst[0]-1, rst[0]-1), mode="constant", value=0)
        qeff = torch.nn.functional.conv3d(qeff, kernel, padding='valid',
                                              groups=lmn_prod)

        qeff = qeff.view(len(charge), lmn[0], lmn[1], lmn[2], shape[0]+1, shape[1]+1, shape[2]+1)
        qeff = torch.sum(qeff, dim=[1, 2, 3])

        return qeff

    def _create_w1ds(self):
        '''1D weights with the size (L, M, N) for x, y, z
        '''
        return QEff3D.create_w1ds(self._method, self._npoints, self._grid_spacing)

    def _create_u1ds(self):
        '''1D weights for trilinear interpolation
            with the size (L, M, N, 2, 2, 2) for x, y, z
        '''
        return QEff3D.create_u1ds(self._method, self._npoints)


    def _create_sampling_1d(self):
        '''create grid in 1d with a size of (L, I), (M, ).
        Note:
        1. A uniform spacing of cubes is assumed.
        2. (b-a)/2 of each interval is included in the weight
        '''
        sampling_1ds = QEff3D.create_sampling_1ds(self._method, self._origin, self._grid_spacing, self._box_offset, self._box_shape, self._npoints)
        return sampling_1ds

    def create_qeff_noweight(self, Q, X0, X1, Sigma, x, y, z):
        '''eval Q on sampling points without weights
        Q is scalar,
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
        '''
        return QEff3D.eval_line_conv_gaus(Q, X0, X1, Sigma, x, y, z)

    def create_qeff(self, Q, X0, X1, Sigma, **kwargs):
        '''create Qeff multiplied by weights; output is squeezed'''
        qeff = QEff3D.eval_qeff(Q, X0, X1, Sigma, self._box_offset, self._box_shape,
                                self._origin, self._grid_spacing, self._method, self._npoints,
                                **kwargs)
        return qeff