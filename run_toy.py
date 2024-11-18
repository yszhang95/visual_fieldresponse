import toy
from toy import Qeff

import numpy as np

import tracemalloc

import timeit

import logging

logger = logging.getLogger(__name__)

def run_test_case(x, y, t, meshgrid, f=None):
    qeff = Qeff(x, y, t, meshgrid=meshgrid, method='gauss_legendre_4')
    if f is None:
        qeff.func = lambda x, y, t : x**3 * y**3 * t**3
    else:
        qeff.func = f
    charge_eff = qeff.create_qeff()
    if f is None:
        integral = lambda x0, x1, y0, y1, t0, t1 : \
            (x1**4-x0**4)*(y1**4-y0**4)*(t1**4-t0**4)/4**3
        logger.info("Analytical integral value of x**3 * y**3 * t**3 for the "\
              "interval is", integral(x[0], x[1], y[0], y[1], t[0], t[1]))
    logger.info('Numerical integral value is {}'.format(np.sum(charge_eff)))

def memory_stats(meshgrid=True):
    '''make a statistics of memory usage'''
    print('========== memory_stats starts ==========')
    tracemalloc.start()
    run_test_case(x=(0,10,101), y=(0,10,101), t=(0,10,101),
                  meshgrid=meshgrid, f=None)

    snapshot2 = tracemalloc.take_snapshot()

    top_stats = snapshot2.statistics('traceback')

    print("[ Top 5 memory] for 101x101x101 grids using meshgrid={}"\
          .format(meshgrid))
    for stat in top_stats[:5]:
       print(stat)
    tracemalloc.stop()
    print('======= memory_stats ends ===========\n')

def runtime_analysis(meshgrid=True, number=100):
    x=(0,10,51)
    y=(0,10,51)
    t=(0,10,51)
    runstr = 'run_test_case(x={},y={},t={},meshgrid={})'\
        .format(x, y, t, meshgrid)
    elapsed_time = \
        timeit.timeit(runstr, setup='from __main__ import run_test_case',
                      number=number)
    print('Total time given {} loops: {} seconds'.format(number, elapsed_time))
    print('Average time per loop', elapsed_time/number, 'seconds')

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    memory_stats(True)
    # memory_stats(False)
    runtime_analysis()
