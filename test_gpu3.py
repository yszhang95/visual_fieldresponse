import numpy as np
import scipy as sp
import json

import torch

from toy_gpu3 import UniversalGrid, LocalGrid, QModel, QEff3D

def test_create_w1d_GL(doprint=True):
    # assertion matched
    if doprint:
        print('\nTesting 4-point weights calculation in 1D ----------------')
    # print('4 point weights for interval length of 0.1, QEff3D.create_w1d_GL(4, 0.1))',
    #       QEff3D.create_w1d_GL(4, 0.1))
    # print('4 point weights for interval length of 0.1, scipy.special.roots_legendre(4)[1] * 0.05',
    #       sp.special.roots_legendre(4)[1] * 0.05)
    assert np.all(np.abs(np.array(QEff3D.create_w1d_GL(4, 0.1)) - sp.special.roots_legendre(4)[1] * 0.05)/
                  np.abs(np.array(QEff3D.create_w1d_GL(4, 0.1)))< 1E-5)
    if doprint:
        print('pass assertion at rel. delta < 1E-5')

def test_create_u1d_GL(doprint=True):
    # assertion matched
    if doprint:
        print('\nTesting u calculation in 1D ----------------')
    # print('4 point roots of legendre are', sp.special.roots_legendre(4)[0])
    # print('4 point u = (r+1)/2 for (1-u, u)', QEff3D.create_u1d_GL(4))
    # print('You know, interpolation is f(0) * (1-u) + f(1) * u; when r = -1, u=0, 1-u=1, interpolation == f(0).')
    assert np.all(np.abs(np.array(QEff3D.create_u1d_GL(4)[:,1]) - (sp.special.roots_legendre(4)[0]+1)/2)/np.abs(np.array(QEff3D.create_u1d_GL(4)[:,1]))  < 1E-5)
    assert np.all(np.abs(np.array(QEff3D.create_u1d_GL(4)[:,0]) + np.array(QEff3D.create_u1d_GL(4)[:,1]) - 1) < 1E-5)
    if doprint:
        print('pass assertion at rel. delta < 1E-5')

def test_create_sampling_1d_GL(doprint=True):
    # assertion matched
    if doprint:
        print('\nTesting sampling points in 1D ----------------')
    # print('4 point, GL roots', sp.special.roots_legendre(4)[0], 'transformed by 0.1/2*roots + 0.1/2',
    #      sp.special.roots_legendre(4)[0] * 0.05 + 0.05)
    # print('4 point, sampling points along [0, 1], [2,3], [3,4]',
    #       QEff3D.create_sampling_1d_GL(0, 0.1, torch.tensor([0, 20, 30], dtype=torch.int64), 10, 4))
    for i, iv in enumerate([0, 20, 30]): # batch dimension
        for j in range(4): # n points
            for k in range(10): # 10 intervals
                assert ( np.abs(
                    (sp.special.roots_legendre(4)[0] * 0.05 + 0 + 0.1*iv + (k+0.5)*0.1)[j]
                          - QEff3D.create_sampling_1d_GL(0, 0.1, torch.tensor([0, 20, 30], dtype=torch.int64), 10, 4)[i,j,k]
                    )
                      /np.abs(QEff3D.create_sampling_1d_GL(0, 0.1, torch.tensor([0, 20, 30], dtype=torch.int64), 10, 4)[i,j,k]) < 1E-5 )
    if doprint:
        print('pass assertion at rel. delta < 1E-5')

def test_create_w1ds(doprint=True):
    '''
    create_w1d must be true
    '''
    # assertion match
    if doprint:
        print('\nTesting 4-point weights calculation in 1D for three dimensions, interval width == 0.1 ----------')
    test_create_w1d_GL(False)
    # print(QEff3D.create_w1ds('gauss_legendre', (4,4,4), (0.1, 0.1, 0.1)))
    for i in range(3):
        assert (torch.all(torch.abs(QEff3D.create_w1ds('gauss_legendre', (4,4,4), (0.1, 0.1, 0.1))[i] - QEff3D.create_w1d_GL(4, 0.1))
                      / np.abs(QEff3D.create_w1d_GL(4, 0.1)) < 1E-5))
    if doprint:
        print('pass assertion with rel. delta <1E-5')

def test_create_u1ds(doprint=True):
    '''
    create_u1d_GL must pass assertion
    '''
    # matched expectation; u1d[:, 0] + u1d[:, 1] = 1
    if doprint:
        print('\nTesting 4-point u calculation in 1D for three dimensions ----------')
    test_create_u1d_GL(False)
    # print(QEff3D.create_u1ds('gauss_legendre', (4,4,4)))
    for i, npts in enumerate((4,4,4)):
        for ipt in range(npts):
            assert abs(QEff3D.create_u1ds('gauss_legendre', (4,4,4))[i][ipt,0] - QEff3D.create_u1d_GL(4)[ipt,0])/abs(QEff3D.create_u1d_GL(4)[ipt,0]) < 1E-4
            assert abs(QEff3D.create_u1ds('gauss_legendre', (4,4,4))[i][ipt,1] - QEff3D.create_u1d_GL(4)[ipt,1])/abs(QEff3D.create_u1d_GL(4)[ipt,1]) < 1E-4
    if doprint:
        print('pass assertion with rel. delta <1E-4')

def test_create_weight_block(doprint=True):
    '''
    create_w1ds must pass assertion
    '''
    # matched expectation; asserted
    if doprint:
        print('\nTesting (3,2,1)-point weight block calculation in 3D, interval width == 2 -------')
    test_create_w1ds(False)
    # print(QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (3, 2, 1), (2,2,2))))
    for i in range(3):
        for j in range(2):
            for k in range(1):
                assert abs(
                    QEff3D.create_w1ds('gauss_legendre', (3, 2, 1), (2,2,2))[0][i]
                    * QEff3D.create_w1ds('gauss_legendre', (3, 2, 1), (2,2,2))[1][j]
                    * QEff3D.create_w1ds('gauss_legendre', (3, 2, 1), (2,2,2))[2][k] -
                    QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (3, 2, 1), (2,2,2)))[i,j,k]
                )/abs(QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (3, 2, 1), (2,2,2)))[i,j,k]) < 1E-5
    if doprint:
        print('pass rel assertion with rel. delta <1E-4')

def test_create_u_block(doprint=True):
    '''
    create_u1ds must pass assertion
    '''
    # matched expectation; u3d[0,0,0,0,0,0] = u1d[0,0] (0.5) * u1d[0,0] (0.7887) * u1d[0,0] (0.5), asserted
    if doprint:
        print('\nTesting (1,2,1)-point u block calculation in 3D, -------')
    test_create_u1ds(False)
    # print('Three 1D u arrays', QEff3D.create_u1ds('gauss_legendre', (1, 2, 1)))
    # print(QEff3D.create_u_block(QEff3D.create_u1ds('gauss_legendre', (1, 2, 1)))) #
    for l in range(1):
        for m in range(2):
            for n in range(1):
                for r in range(2):
                    for s in range(2):
                        for t in range(2):
                            assert abs(
                    QEff3D.create_u1ds('gauss_legendre', (1, 2, 1))[0][l,r]
                    * QEff3D.create_u1ds('gauss_legendre', (1, 2, 1))[1][m,s]
                    * QEff3D.create_u1ds('gauss_legendre', (1, 2, 1))[2][n,t]
                    - QEff3D.create_u_block(QEff3D.create_u1ds('gauss_legendre', (1, 2, 1)))[l,m,n,r,s,t]
                ) <1E-5
    if doprint:
        print('pass rel assertion with rel. delta <1E-5')

def test_create_wu_block(doprint=True):
    '''
    create_u_block, create_weight_block must pass assertion
    '''
    # matched expectation
    if doprint:
        print('\nTesting (1,2,1)-point wu block calculation in 3D, interval width == 1 -------')
    test_create_weight_block()
    test_create_u_block()
    # print(QEff3D.create_wu_block(
    #     QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (1, 2, 1), (1,1,1))),
    #     QEff3D.create_u_block(QEff3D.create_u1ds('gauss_legendre', (1, 2, 1)))))
    for l in range(1):
        for m in range(2):
            for n in range(1):
                for r in range(2):
                    for s in range(2):
                        for t in range(2):
                            assert abs(
                                QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (1, 2, 1), (1,1,1)))[l,m,n] *
                                QEff3D.create_u_block(QEff3D.create_u1ds('gauss_legendre', (1, 2, 1)))[l,m,n,r,s,t]
                                - QEff3D.create_wu_block(
                                    QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (1, 2, 1), (1,1,1))),
                                    QEff3D.create_u_block(QEff3D.create_u1ds('gauss_legendre', (1, 2, 1))))[l,m,n,r,s,t]
                            )/abs(QEff3D.create_wu_block(
                                    QEff3D.create_weight_block(QEff3D.create_w1ds('gauss_legendre', (1, 2, 1), (1,1,1))),
                                    QEff3D.create_u_block(QEff3D.create_u1ds('gauss_legendre', (1, 2, 1))))[l,m,n,r,s,t]) < 1E-5
    if doprint:
        print('pass rel assertion with rel. delta <1E-5')

def test_QEff():
    print('\nTesting parse_method, gauss_legendre_2_2_2 ----------------')
    m, npts = QEff3D.parse_method('gauss_legendre_2_2_2')
    print(m, npts)

    test_create_w1d_GL()

    test_create_u1d_GL()

    test_create_sampling_1d_GL()

    test_create_w1ds()

    test_create_u1ds()

    test_create_weight_block()

    test_create_u_block()

    test_create_wu_block()

    print('\nTesting 4-point GL rule using x**3 * y**3 * t**3; Contributions from Q, X0, X1, Sigma, are ignored')
    print('Setup, x in [0,1), y in [2, 3), t in [3, 4)')
    qeff = QEff3D(
        origin=(0,0,0), grid_spacing=(0.1, 0.1, 0.1), offset=[(0,20,30)], shape=(10, 10, 10),
        method='gauss_legendre_4_4_4')

    def mymodel(Q, X0, X1, Sigma, x, y, t):
        output = x**3 * y**3 * t**3
        w1d = QEff3D.create_w1d_GL(4, 0.1)
        xw = x**3 * w1d.view(1, 4, 1, 1, 1, 1, 1)
        yw = y**3 * w1d.view(1, 1, 4, 1, 1, 1, 1)
        tw = t**3 * w1d.view(1, 1, 1, 4, 1, 1, 1)
        print('in my model, x, y, t shape', x.shape, y.shape, t.shape)
        print('in my model, output shape', output.shape)
        print('in my model, xw shape', xw.shape)
        print('in my model, x*w1d sum', torch.sum(xw))
        print('in my model, y*w1d sum', torch.sum(yw))
        print('in my model, t*w1d sum', torch.sum(tw))
        print('in my model, xyt * w3d sum', torch.sum(output * w1d[None, :, None, None, None, None, None]
                                  * w1d[None, None, :, None, None, None, None]
                                  * w1d[None, None, None, :, None, None, None]))

        return output
    # qmodel = lambda Q, X0, X1, Sigma, x, y, t :
    dummy = torch.tensor([1], device='cpu')
    effq = qeff.create_qeff(None, dummy, None, None, qmodel=mymodel)
    print('effective charge calculation, shape', effq.shape, 'total Q', torch.sum(effq))
    assert abs(torch.sum(effq).item() - 177.7344)/177.7344 < 1E-4
    print('Pass assertion for x^3 * y^3 * t^3')

    ilinear = lambda x, y, t : x * y * t
    x = np.linspace(0, 1, 11)
    y = np.linspace(2, 3, 11)
    t = np.linspace(3, 4, 11)
    xgrid, ygrid, tgrid = np.meshgrid(x, y, t, indexing='ij')
    I = torch.tensor(ilinear(xgrid, ygrid, tgrid), dtype=torch.float32)
    Y = effq
    assert abs((torch.sum(Y * I) - 1318.3282).item())/1318.3282 <1E-5
    print('Pass assertion for x^4 * y^4 * t^4 after multiplying linear model x*y*t')

    # Asserted
    print('\nTesting line conv gaus model in QEff3D-------------')
    Q=(1,)
    X0=[(0.4,2.4,3.4)]
    X1=[(0.6, 2.6, 3.6)]
    Sigma=[(0.5, 0.5, 0.5)]
    print(f'Setup, Q={Q}, X0={X0}, X1={X1}, Sigma={Sigma}, Origin={qeff.origin}, GridSpacing={qeff.grid_spacing}, Offset={qeff.box_offset}, Shape={qeff.box_shape}')
    X0 = torch.tensor(X0)
    effq2 = qeff.create_qeff(Q=Q, X0=X0, X1=X1,
                             Sigma=Sigma, qmodel=QModel.GaussConvLine3D)
    print('Sum of Line conv Gaus', torch.sum(effq2))
    assert(np.abs(torch.sum(effq2).item() - 0.3137) < 1E-4) # this value is known... Test breaks for other setup...
    print('Pass assertion for GausConvLine')

def test_QModel():
    X0=(0.4,2.4,3.4)
    X1=(0.6, 2.6, 3.6)
    Sigma=(0.05, 0.05, 0.05)

    x = torch.linspace(0.2, 0.8, 5, dtype=torch.float64).view(1,-1)
    y = torch.linspace(2.2, 2.8, 5, dtype=torch.float64).view(1,-1)
    t = torch.linspace(3.2, 3.8, 5, dtype=torch.float64).view(1,-1)

    testq = QModel.GaussConvLine3D([1], [X0], [X1], [Sigma], x.unsqueeze(2).unsqueeze(2), y.unsqueeze(2).unsqueeze(1), t.unsqueeze(1).unsqueeze(1))
    # testq = QModel.GaussConvLine3D([1], [X0], [X1], [Sigma], x[:,:,None,None], y[:,None,:,None], t[:,None,None,:])
    with open('exact.json') as f:
        exact = json.load(f)
    for i in range(testq.shape[1]):
        for j in range(testq.shape[2]):
            for k in range(testq.shape[3]):
                d = np.abs(testq[0,i,j,k].item()-exact[i][j][k])
                if d > 1E-5:
                    print('difference =', d,'> 1E-5', 'ijk',i, j, k,
                          'x,y,z',x[0,i].item(), y[0,j].item(), t[0,k].item(),
                          'testq', testq[0,i,j,k].item(), 'mathmetica', exact[i][j][k],  'difference', d)

def test_localgrid(origin=(-1,-1,-1), grid_spacing=(1,1,1),n_sigma=(5,3,1)):

    # Initialize the grid
    local_grid = LocalGrid(origin, grid_spacing, n_sigma)

    # Input data
    X0 = torch.tensor([[1.0, 1.0, 1.0], [6, 2.0, 2.0]], dtype=torch.float32)  # Starting points
    X1 = torch.tensor([[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]], dtype=torch.float32)  # Ending points
    Sigma = torch.tensor([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]], dtype=torch.float32)  # Diffusion widths
    n_sigma = torch.tensor(n_sigma, dtype=torch.float32)
    print('Input X0 {}\nX1 {}\nSigma {}'.format(X0, X1, Sigma))

    # Compute charge box
    result = local_grid.get_charge_box(X0, X1, Sigma)
    print('from get_charge_box')
    print("Offsets:", result[0])
    print("Shapes:", result[1])
    result = LocalGrid.compute_charge_box(X0, X1, Sigma,n_sigma, origin, grid_spacing)
    print('from compute_charge_box')
    print('Offsets:', result[0])
    print('Shapes:', result[1])
    print("Bounds of with origin {}, spacing {}, n_sigma {}".format(origin, grid_spacing, n_sigma), local_grid.compute_bounds(X0, X1, Sigma))
    print("Bounds of with origin {}, spacing {}, n_sigma {}, from static,".format(origin, grid_spacing, n_sigma), LocalGrid.compute_bounds_X0_X1(X0, X1, Sigma, torch.tensor(n_sigma)))
    print('Stack of X0, X1', LocalGrid.stack_X0X1(X0, X1))
    print("Bounds of with origin {}, spacing {}, n_sigma {}, after stacking and static,".format(origin, grid_spacing, n_sigma), LocalGrid.compute_bounds_X0X1(LocalGrid.stack_X0X1(X0, X1), Sigma, n_sigma))

def print_grid(grid):

    # Get physical coordinates for given indices
    coords = grid.get_coordinate([(2,3,4),(3,4,5)])
    print("Coordinates for indices[(2,3,4),(3,4,5)]:", coords)
    coords = grid.get_coordinate([2, 3, 4])
    print("Coordinates for indices (2, 3, 4):", coords)

    # Convert coordinates to indices
    physical_coords = torch.tensor([[2.5, 3.5, 1.5], [7.0, 8.0, 9.0]], dtype=torch.float32)
    indices = grid.get_index(physical_coords)
    print("Indices for physical coordinates {}:".format(physical_coords), indices)
    physical_coords = torch.tensor([2.5, 3.5, 1.5], dtype=torch.float32)
    indices = grid.get_index(physical_coords)
    print("Indices for physical coordinates {}:".format(physical_coords), indices)

    print('Indices of {} from computation'.format([[2.5, 3.5, 1.5], [-6.9, 8.0, 9.0]]), UniversalGrid.compute_index([[2.5, 3.5, 1.5], [-6.9, 8.0, 9.0]], origin=grid.origin, spacing=grid.grid_spacing))
    print('Coordinates from computation', UniversalGrid.compute_coordinate([[-2,3,4], [3,4,5]], origin=grid.origin, spacing=grid.grid_spacing))

def test_grid():
    # Define grid parameters
    origin = (0.0, 0.0, 0.0)
    grid_spacing = (1.0, 1.0, 1.0)
    print('setup')
    print('origin', origin)
    print('spacing', grid_spacing)
    # Initialize separated grid
    grid = UniversalGrid(grid_spacing, origin)
    print_grid(grid)
    print('-----')
    grid2 = UniversalGrid().from_grid(grid.grid_spacing, grid.origin)
    # print(grid2.get_coord([2,3,4], offset=None))
    print_grid(grid2)
    print('----- coarse grain', [2,1,1])
    grid3 = grid2.coarse_grain([2,1,1])
    print_grid(grid3)


def test_cond_QEff():
    device = 'cuda:0'
    qeff = QEff3D(
        origin=(0,0,0), grid_spacing=(0.1, 0.1, 0.1), offset=[(0,20,30)], shape=(10, 10, 10),
        method='gauss_legendre_4_4_4', device=device)

    # Asserted
    print('\nTesting line conv gaus model in QEff3D-------------')
    Q=(1,)
    X0=[(0.4,2.4,3.4)]
    X1=[(0.6, 2.6, 3.6)]
    Sigma=[(0.5, 0.5, 0.5)]
    print(f'Setup, Q={Q}, X0={X0}, X1={X1}, Sigma={Sigma}, Origin={qeff.origin}, GridSpacing={qeff.grid_spacing}, Offset={qeff.box_offset}, Shape={qeff.box_shape}')
    Q = torch.tensor(Q, device=device)
    X0 = torch.tensor(X0, device=device)
    X1 = torch.tensor(X1, device=device)
    Sigma = torch.tensor(Sigma, device=device)
    effq = qeff.create_qeff(Q=Q, X0=X0, X1=X1,
                             Sigma=Sigma, qmodel=QModel.GaussConvLine3D,
                             usemask=True, n_sigma=(3.5, 3.5, 3.5))

    effq2 = qeff.create_qeff(Q=Q, X0=X0, X1=X1,
                             Sigma=Sigma, qmodel=QModel.GaussConvLine3D)
    print('Sum of Line conv Gaus', torch.sum(effq))
    print('Sum of Line conv Gaus', torch.sum(effq2))
    assert(np.abs(torch.sum(effq2).item() - 0.3137) < 1E-4) # this value is known... Test breaks for other setup...
    print('Pass assertion for GausConvLine')

if __name__ == '__main__':
    test_QModel()
    test_localgrid()
    test_grid()
    test_QEff()
    test_cond_QEff()
