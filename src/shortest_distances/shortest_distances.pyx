# cython: language_level=3
import time

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport numpy


def find_shortest_distances(numpy.ndarray[double, ndim=2] friction_array):
    """Define later

    Parameters:
        friction_array (numpy.ndarray): array with friction values for
            determining lcp.

    Returns:
        ?

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

    cdef double cell_length = 1.0
    cdef double diagonal_cell_length = 2**0.5

    # local cell numbering scheme
    # 0 1 2
    # 3 x 4
    # 5 6 7
    # there's symmetry in this structure since it's fully connected so
    # on each element we connect to elements 2, 4, 6, and 7 only
    for i in range(win_xsize):
        for j in range(win_ysize):
            center_val = friction_array[j, i]
            flat_index = j*win_xsize+i
            if j > 0 and i < win_xsize - 1:
                diagonals[2][flat_index-win_xsize+1] = (
                    diagonal_cell_length * (
                        friction_array[j-1, i+1] + center_val) / 2.0)
            if i < win_xsize - 1:
                diagonals[4][flat_index+1] = (
                    cell_length * (
                        friction_array[j, i+1] + center_val) / 2.0)
            if j < win_ysize-1:
                diagonals[6][flat_index+win_xsize] = (
                    cell_length * (
                        friction_array[j+1, i] + center_val) / 2.0)
            if j < win_ysize-1 and i < win_xsize - 1:
                diagonals[7][flat_index+win_xsize+1] = (
                    diagonal_cell_length * (
                        friction_array[j+1, i+1] + center_val) / 2.0)

    dist_matrix = scipy.sparse.dia_matrix((
        diagonals, diagonal_offsets), shape=(n, n))
    # numpy.set_printoptions(
    #     threshold=numpy.inf, linewidth=numpy.inf, precision=3)
    # print(dist_matrix.toarray())
    print('calculate distances')
    distances = scipy.sparse.csgraph.shortest_path(
        dist_matrix, method='auto', directed=False)
    print('total time on %d elements: %s', win_xsize, time.time() - start_time)
    print(distances)
    """

    dist_matrix = scipy.sparse.csc_matrix(
        (([1], ([0], [1]))), (array.size, array.size))
    print('making distances')
    start_time = time.time()
    distances = scipy.sparse.csgraph.shortest_path(
        dist_matrix, method='auto', directed=False)
    print(distances)
    print('total time: %s', time.time() - start_time)
    """
