# cython: language_level=3
import time
import logging
import sys
import heapq

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport numpy
from libc.time cimport time as ctime
from libc.time cimport time_t

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def find_population_reach(
        numpy.ndarray[float, ndim=2] friction_array,
        numpy.ndarray[int, ndim=2] population_array,
        double cell_length_m, int core_i, int core_j,
        int core_size_i, int core_size_j,
        int n_cols, int n_rows,
        double max_time):
    """Define later

    Parameters:
        friction_array (numpy.ndarray): array with friction values for
            determining lcp in units minutes/pixel.
        population_array (numpy.ndarray): array with population values per
            pixel.
        cell_length_m (double): length of cell in meters.
        core_i/core_j (int): defines the ul corner of the core in
            arrays.
        core_size_i/j (int): defines the w/h of the core slice in
            arrays.
        n_cols/n_rows (int): number of cells in i/j direction of given arrays.
        max_time (double): the time allowed when computing population reach
            in minutes.

    Returns:
        core_size 2D array of population reach starting at core_x/y on the
        friction array.

    """
    start_time = time.time()
    cdef int i, j
    cdef numpy.ndarray[double, ndim=2] pop_coverage = numpy.zeros(
        (n_rows, n_cols))
    cdef numpy.ndarray[numpy.npy_bool, ndim=2] visited

    LOGGER.debug(
        f'core_i, core_j: {core_i},{core_j}\n'
        f'core_size_i, core_size_j: {core_size_i},{core_size_j}\n'
        f'n_rows, n_cols: {n_rows},{n_cols}')

    cdef list time_heap = []
    cdef int *ioff = [1, 1, 0, -1, -1, -1, 0, 1]
    cdef int *joff = [0, 1, 1, 1, 0, -1, -1, -1]
    cdef float *dist_edge = [
        cell_length_m,
        cell_length_m*2**0.5,
        cell_length_m,
        cell_length_m*2**0.5,
        cell_length_m,
        cell_length_m*2**0.5,
        cell_length_m,
        cell_length_m*2**0.5]
    cdef float frict_n, c_time, n_time,
    cdef int i_start, j_start, i_n, j_n, n_steps
    cdef time_t last_log_time

    for i_start in range(core_i, core_i+core_size_i):
        for j_start in range(core_j, core_j+core_size_j):
            population_val = population_array[j_start, i_start]
            if population_val <= 0:
                continue
            visited = numpy.zeros((n_rows, n_cols), dtype=bool)
            time_heap = [(0, (j_start, i_start))]

            # c_ -- current, n_ -- neighbor
            last_log_time = ctime(NULL)
            n_steps = 0
            while time_heap:
                c_time, (j, i) = heapq.heappop(time_heap)
                visited[j, i] = True
                n_steps += 1
                if ctime(NULL) - last_log_time > 5.0:
                    last_log_time = ctime(NULL)
                    LOGGER.info(
                        f'Processing {j_start}, {i_start} for {n_steps}')

                for v in range(8):
                    i_n = i+ioff[v]
                    j_n = j+joff[v]
                    if i_n < 0 or i_n >= n_cols:
                        continue
                    if j_n < 0 or j_n >= n_rows:
                        continue
                    if visited[j_n, i_n]:
                        continue
                    frict_n = friction_array[j_n, i_n]
                    if frict_n <= 0:
                        continue
                    n_time = c_time + frict_n*dist_edge[v]
                    if n_time <= max_time:
                        heapq.heappush(time_heap, (n_time, (j_n, i_n)))
            pop_coverage += population_val * visited
    LOGGER.debug(
        f'completed ({core_i}, {core_j}) in {time.time()-start_time}s')
    return pop_coverage
