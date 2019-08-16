# cython: language_level=3
import time

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
    cdef double[:, :] diagonals = numpy.empty((n, 9))
    cdef double[:, :] raster_array
    cdef int i, j

    raster = gdal.OpenEx(raster_path_band[0])
    band = raster.GetRasterBand(raster_path_band[1])
    print('opening %s' % str(raster_path_band))
    raster_array = band.ReadAsArray(
        xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)

    for i in range(win_xsize):
        for j in range(win_ysize):
            pass

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
