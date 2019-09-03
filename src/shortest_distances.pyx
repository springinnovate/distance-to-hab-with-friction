# cython: language_level=3
import time

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport numpy

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def find_population_reach(
        numpy.ndarray[double, ndim=2] friction_array,
        numpy.ndarray[double, ndim=2] population_array,
        double cell_length, int core_x, int core_y, int core_size,
        double max_time, double max_travel_distance):
    """Define later

    Parameters:
        friction_array (numpy.ndarray): array with friction values for
            determining lcp in units minutes/meter.
        population_array (numpy.ndarray): array with population values per
            pixel.
        cell_length (double): length of cell in meters.
        core_x/core_y (int): defines the ul corner of the core in friction
            array
        core_size (int): defines the w/h of the core slice in friction_array.
        max_time (double): the time allowed when computing population reach.
        max_travel_distance (double): the maximum distance allowed to travel
             in meters.

    Returns:
        core_size 2D array of population reach starting at core_x/y on the
        friction array.

    """
    start_time = time.time()
    cdef int win_xsize, win_ysize
    win_xsize = friction_array.shape[1]
    win_ysize = friction_array.shape[0]
    cdef int n = win_xsize * win_ysize
    cdef int flat_index
    cdef double center_val
    cdef double[:, :] diagonals = numpy.zeros((4, n))
    cdef int[:] diagonal_offsets = numpy.array([
        -win_xsize, -win_xsize+1, 1, win_xsize+1], dtype=numpy.int32)
    cdef int i, j
    cdef double diagonal_cell_length = 2**0.5 * cell_length
    cdef double working_val

    # local cell numbering scheme
    # 0 1 2
    # 3 x 4
    # 5 6 7
    # there's symmetry in this structure since it's fully connected so
    # on each element we connect to elements 2, 4, 6, and 7 only
    LOGGER.debug('build distance array')
    cdef int n_elements = 0
    for i in range(win_xsize):
        for j in range(win_ysize):
            if i == j:
                continue
            center_val = friction_array[j, i]
            if center_val != center_val:  # it's NaN so skip
                continue
            n_elements += 1
            flat_index = j*win_xsize+i
            if j > 0:
                working_val = friction_array[j-1, i]
                if working_val == working_val:
                    diagonals[0][flat_index-win_xsize] = (
                        diagonal_cell_length * (
                            working_val + center_val) / 2.0)
            if j > 0 and i < win_xsize - 1:
                working_val = friction_array[j-1, i+1]
                if working_val == working_val:
                    diagonals[1][flat_index-win_xsize+1] = (
                        diagonal_cell_length * (
                            working_val + center_val) / 2.0)
            if i < win_xsize - 1:
                working_val = friction_array[j, i+1]
                if working_val == working_val:
                    diagonals[2][flat_index+1] = (
                        cell_length * (
                            working_val + center_val) / 2.0)
            if j < win_ysize-1 and i < win_xsize - 1:
                working_val = friction_array[j+1, i+1]
                if working_val == working_val:
                    diagonals[3][flat_index+win_xsize+1] = (
                        diagonal_cell_length * (
                            working_val + center_val) / 2.0)

    dist_matrix = scipy.sparse.dia_matrix((
        diagonals, diagonal_offsets), shape=(n, n))
    LOGGER.debug('calculate distances')
    cdef numpy.ndarray[double, ndim=2] travel_time
    cdef numpy.ndarray[int, ndim=2] predecessors
    travel_time, predecessors = (
        scipy.sparse.csgraph.shortest_path(
            dist_matrix, method='auto', directed=False,
            return_predecessors=True))
    LOGGER.debug('total time on %d elements: %s' % (
        n_elements, time.time() - start_time))

    cdef numpy.ndarray[double, ndim=2] population_reach = numpy.zeros(
        (core_size, core_size))
    cdef int core_flat_index, core_i, core_j, n_steps, current_node
    cdef double population_count
    start_time = time.time()
    LOGGER.debug(
        'distance percentiles: %s' % numpy.percentile(
            travel_time[travel_time != numpy.inf], [0, 25, 50, 75, 100]))
    cdef int max_travel_steps = int(max_travel_distance / cell_length)
    for core_i in range(core_size):
        for core_j in range(core_size):
            # the core x/y starts halfway in on the cor length of the raster
            # window, so adding those in directly into the flat index.
            core_flat_index = core_i + core_x + (
                core_j + core_y)*win_xsize
            population_count = 0.0
            for i in range(win_xsize):
                for j in range(win_ysize):
                    flat_index = j*win_xsize+i
                    if flat_index == core_flat_index:
                        population_count += population_array[j, i]
                    else:
                        if travel_time[core_flat_index, flat_index] < max_time:
                            # walk the predecessor array
                            n_steps = 0
                            current_node = core_flat_index
                            while current_node != flat_index:
                                current_node = (
                                    predecessors[flat_index, current_node])
                                n_steps += 1
                                if n_steps > max_travel_steps:
                                    break
                            if n_steps <= max_travel_steps:
                                population_count += population_array[j, i]
            population_reach[core_j, core_i] = population_count

    LOGGER.debug(
        'total time on determining pop elements %.2fs' % (
            time.time() - start_time))
    return population_reach
