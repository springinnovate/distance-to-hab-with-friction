# cython: language_level=3
import time

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport numpy


def find_shortest_distances(
        raster_path_band, xoff, yoff, win_xsize, win_ysize):
    """Find shortest distances in a subgrid of raster_path.

    Parameters:
        raster_path_band (tuple): raster path band index tuple.
        xoff, yoff, xwin_size, ywin_size (int): rectangle that defines
            upper left hand corner and size of a subgrid to extract from
            ``raster_path_band``.
        edge_buffer_elements (int): number of pixels to buffer around the
            defined rectangle for searching distances.

    Returns:
        ?

    """
    cdef int n = win_xsize*win_ysize
    cdef double[:, :] diagonals = numpy.empty((8, n))
    cdef int[:] diagonal_offsets = numpy.array([
        -win_xsize-1, -win_xsize, -win_xsize+1, -1, 1,
        win_xsize-1, win_xsize, win_xsize+1], dtype=numpy.int)
    cdef double[:, :] raster_array
    cdef int i, j

    raster = gdal.OpenEx(raster_path_band[0])
    band = raster.GetRasterBand(raster_path_band[1])
    print('opening %s' % str(raster_path_band))

    cdef double cell_length = abs(
        pygeoprocessing.get_raster_info(
            raster_path_band[0])['pixel_size'][0])
    cdef double diagonal_cell_length = (2**0.5)*cell_length

    raster_array = band.ReadAsArray(
        xoff=xoff, yoff=yoff, win_xsize=win_xsize,
        win_ysize=win_ysize).astype(numpy.double)

    # local cell numbering scheme
    # 0 1 2
    # 3 x 4
    # 5 6 7

    for i in range(win_xsize):
        for j in range(win_ysize):
            center_val = raster_array[j, i]
            flat_index = j*win_xsize+i
            if i > 0 and j > 0:
                diagonals[0][diagonal_offsets[0] + flat_index-win_xsize-1] = (
                    diagonal_cell_length * (
                        raster_array[j-1, i-1] + center_val) / 2.0)
            if j > 0:
                diagonals[1][diagonal_offsets[0] + flat_index-win_xsize] = (
                    cell_length * (
                        raster_array[j-1, i] + center_val) / 2.0)
            if j > 0 and i < win_xsize - 1:
                diagonals[2][diagonal_offsets[0] + flat_index-win_xsize+1] = (
                    diagonal_cell_length * (
                        raster_array[j-1, i+1] + center_val) / 2.0)
            if i > 0:
                diagonals[3][diagonal_offsets[0] + flat_index-1] = (
                    cell_length * (
                        raster_array[j, i-1] + center_val) / 2.0)
            if i < win_xsize - 1:
                diagonals[4][diagonal_offsets[0] + flat_index+1] = (
                    cell_length * (
                        raster_array[j, i+1] + center_val) / 2.0)
            if j < win_ysize-1 and i > 0:
                diagonals[5][diagonal_offsets[0] + flat_index+win_xsize-1] = (
                    diagonal_cell_length * (
                        raster_array[j+1, i-1] + center_val) / 2.0)
            if j < win_ysize-1:
                diagonals[6][diagonal_offsets[0] + flat_index+win_xsize] = (
                    cell_length * (
                        raster_array[j+1, i] + center_val) / 2.0)
            if j < win_ysize-1 and i < win_xsize - 1:
                diagonals[7][diagonal_offsets[0] + flat_index+win_xsize+1] = (
                    diagonal_cell_length * (
                        raster_array[j+1, i+1] + center_val) / 2.0)
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
