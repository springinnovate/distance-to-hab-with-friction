# cython: language_level=3
import time

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport numpy

shortest_distances.find_shortest_distances(
                friction_raster_path, population_raster_path,
                target_people_access_path,
                raster_i, local_i, buffer_xsize,
                raster_j, local_j, buffer_ysize,
                (window_size + 2*buffer_size, window_size + 2*buffer_size))


def find_shortest_distances(
        friction_raster_path, population_raster_path, aggregate_raster_path,
        raster_xoff, raster_xsize, buffer_i, buffer_xsize, window_i, window_i_buffer_size,
        raster_yoff, raster_ysize, buffer_j, buffer_ysize, window_j, window_j_buffer_size,
        buffer_shape):
    """Define later

    Parameters:
        friction_raster_path (str): friction in m/min of movement through a
            pixel
        population_raster_path (str): population count per pixel
        aggregate_raster_path (str): modified by this call in the window of
            raster_xoff/raster_yoff -> buffer_xsize/buffer_ysize.
        raster_xoff/yoff (int): offset to start reading friction layer.
        buffer_xsize/buffer_ysize (int): xy/window sizes to read from friction
            layer
        buffer_i/buffer_j (int): i/j location that will contain valid
            source points.
        window_i/window_j (int):
        window_i_buffer_size
        window_j_buffer_size



        raster_path_band (tuple): raster path band index tuple.
        xoff, yoff, xwin_size, ywin_size (int): rectangle that defines
            upper left hand corner and size of a subgrid to extract from
            ``raster_path_band``.
        edge_buffer_elements (int): number of pixels to buffer around the
            defined rectangle for searching distances.

    Returns:
        ?

    """
    start_time = time.time()
    cdef int n = win_xsize*win_ysize
    cdef double[:, :] diagonals = numpy.zeros((8, n))
    cdef int[:] diagonal_offsets = numpy.array([
        -win_xsize-1, -win_xsize, -win_xsize+1, -1, 1,
        win_xsize-1, win_xsize, win_xsize+1], dtype=numpy.int)
    cdef double[:, :] raster_array
    cdef int i, j
    raster = gdal.OpenEx(raster_path_band[0])
    band = raster.GetRasterBand(raster_path_band[1])
    print('opening %s' % str(raster_path_band))

    cdef double cell_length = 1.0
    cdef double diagonal_cell_length = 2**0.5

    raster_array = band.ReadAsArray(
        xoff=xoff, yoff=yoff, win_xsize=win_xsize,
        win_ysize=win_ysize).astype(numpy.double)

    print(n)
    print(numpy.array(diagonals))
    print(numpy.array(raster_array))
    # local cell numbering scheme
    # 0 1 2
    # 3 x 4
    # 5 6 7
    # there's symmetry in this structure since it's fully connected so
    # on each element we connect to elements 2, 4, 6, and 7 only
    for i in range(win_xsize):
        for j in range(win_ysize):
            center_val = raster_array[j, i]
            flat_index = j*win_xsize+i
            if j > 0 and i < win_xsize - 1:
                diagonals[2][flat_index-win_xsize+1] = (
                    diagonal_cell_length * (
                        raster_array[j-1, i+1] + center_val) / 2.0)
            if i < win_xsize - 1:
                diagonals[4][flat_index+1] = (
                    cell_length * (
                        raster_array[j, i+1] + center_val) / 2.0)
            if j < win_ysize-1:
                diagonals[6][flat_index+win_xsize] = (
                    cell_length * (
                        raster_array[j+1, i] + center_val) / 2.0)
            if j < win_ysize-1 and i < win_xsize - 1:
                diagonals[7][flat_index+win_xsize+1] = (
                    diagonal_cell_length * (
                        raster_array[j+1, i+1] + center_val) / 2.0)

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
