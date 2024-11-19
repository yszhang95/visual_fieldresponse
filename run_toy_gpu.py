from toy_gpu import QModel, Qeff

from scipy import integrate
import numpy as np

import torch

import matplotlib
import matplotlib.pyplot

import logging

logger = logging.getLogger(__name__)

def test_accuracy_cpu():
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

    flog = open('scipy_quad_cpu.txt', 'w')

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
                            f = QModel.create_QModel((x0,y0,t0), (x1,y1,t1), (sx,sy,st))
                            q, e = integrate.tplquad(f, boundaries[0][0], boundaries[0][1],
                                              boundaries[1][0], boundaries[1][1],
                                              boundaries[2][0], boundaries[2][1])
                            logger.debug('process {}, {}, {}, {}, {}'.format((x0, y0, t0),(x1, y1, t1), (sx, sy, st), q, e))
                            flog.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format(
                                x0,y0,t0, x1,y1,t1, sx,sy,st, q, e
                            ))
    flog.close()

def test_accuracy_gpu():
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

    flog = open('gaus_quad_4points_gpu.txt', 'w')

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
                            f = QModel.create_QModel((x0,y0,t0), (x1,y1,t1), (sx,sy,st))
                            qeff.func = f
                            val = torch.sum(qeff.create_qeff())
                            q = val.item()
                            e = 0
                            flog.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format(
                                x0,y0,t0, x1,y1,t1, sx,sy,st, q, e
                            ))
    flog.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.WARNING)
    test_accuracy_gpu()
    logger.setLevel(logging.DEBUG)
    test_accuracy_cpu()
