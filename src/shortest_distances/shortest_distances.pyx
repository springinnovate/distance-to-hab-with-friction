# cython: language_level=3
import time

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport numpy


def find_population_reach(
        numpy.ndarray[double, ndim=2] friction_array,
        numpy.ndarray[double, ndim=2] population_array,
        double cell_length, int core_x, int core_y, int core_size,
        double max_time):
    """Define later

    Parameters:
        friction_array (numpy.ndarray): array with friction values for
            determining lcp in units m/meter.
        population_array (numpy.ndarray): array with population values per
            pixel.
        cell_length (double): length of cell in meters.
        core_x/core_y (int): defines the ul corner of the core in friction
            array
        core_size (int): defines the w/h of the core slice in friction_array.
        max_time (double): the time allowed when computing population reach.

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
    cdef double[:, :] diagonals = numpy.zeros((8, n))
    cdef int[:] diagonal_offsets = numpy.array([
        -win_xsize-1, -win_xsize, -win_xsize+1, -1, 1,
        win_xsize-1, win_xsize, win_xsize+1], dtype=numpy.int)
    cdef int i, j
    cdef double diagonal_cell_length = 2**0.5 * cell_length

    # local cell numbering scheme
    # 0 1 2
    # 3 x 4
    # 5 6 7
    # there's symmetry in this structure since it's fully connected so
    # on each element we connect to elements 2, 4, 6, and 7 only
    for i in range(win_xsize):
        for j in range(win_ysize):
            if i == j:
                continue
            center_val = friction_array[j, i]
            if center_val != center_val:  # it's NaN so skip
                continue
            flat_index = j*win_xsize+i
            if j > 0 and i < win_xsize - 1:
                working_val = friction_array[j-1, i+1]
                if working_val == working_val:
                    diagonals[2][flat_index-win_xsize+1] = (
                        diagonal_cell_length / (
                            working_val + center_val) / 2.0)
            if i < win_xsize - 1:
                working_val = friction_array[j, i+1]
                if working_val == working_val:
                    diagonals[4][flat_index+1] = (
                        cell_length / (
                            working_val + center_val) / 2.0)
            if j < win_ysize-1:
                working_val = friction_array[j+1, i]
                if working_val == working_val:
                    diagonals[6][flat_index+win_xsize] = (
                        cell_length / (
                            working_val + center_val) / 2.0)
            if j < win_ysize-1 and i < win_xsize - 1:
                working_val = friction_array[j+1, i+1]
                if working_val == working_val:
                    diagonals[7][flat_index+win_xsize+1] = (
                        diagonal_cell_length / (
                            working_val + center_val) / 2.0)

    dist_matrix = scipy.sparse.dia_matrix((
        diagonals, diagonal_offsets), shape=(n, n))
    # numpy.set_printoptions(
    #     threshold=numpy.inf, linewidth=numpy.inf, precision=3)
    # print(dist_matrix.toarray())
    print('calculate distances')
    cdef numpy.ndarray[double, ndim=2] travel_time = (
        scipy.sparse.csgraph.shortest_path(
            dist_matrix, method='D', directed=False))
    print('total time on %d elements: %s', win_xsize, time.time() - start_time)
    print(travel_time)

    cdef numpy.ndarray[double, ndim=2] population_reach = numpy.zeros(
        (core_size, core_size))
    cdef int core_flat_index
    cdef double population_count
    start_time = time.time()
    print('calculating population count %d' % core_x)
    print('unique population array: %s' % numpy.unique(population_array))
    print('distance min max: %s %s' % (numpy.max(travel_time[travel_time != numpy.inf]), numpy.min(travel_time)))
    for core_i in range(core_size):
        for core_j in range(core_size):
            # the core x/y starts halfway in on the cor length of the raster
            # window, so adding those in directly into the flat index.
            core_flat_index = core_i+core_size//2 + (
                core_j+core_size//2)*win_xsize
            population_count = 0.0
            for i in range(win_xsize):
                for j in range(win_ysize):
                    flat_index = j*win_xsize+i
                    if flat_index == core_flat_index:
                        population_count += population_array[j, i]
                    else:
                        if travel_time[core_flat_index, flat_index] < max_time:
                            population_count += population_array[j, i]
            #print('population count ij: %f %d %d' % (population_count, i, j))
            population_reach[core_j, core_i] = population_count

    print(
        'total time on determining pop elements %s', time.time() - start_time)
    return population_reach
