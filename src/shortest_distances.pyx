# cython: language_level=3
# distutils: language=c++
import time
import logging
import sys
import heapq

import pygeoprocessing
from osgeo import gdal
import scipy.sparse.csgraph
import numpy

cimport cython
cimport numpy
from libc.time cimport time as ctime
from libc.time cimport time_t
from libcpp.deque cimport deque

# exposing stl::priority_queue so we can have all 3 template arguments so
# we can pass a different Compare functor
cdef extern from "<queue>" namespace "std" nogil:
    cdef cppclass priority_queue[T, Container, Compare]:
        priority_queue() except +
        priority_queue(priority_queue&) except +
        priority_queue(Container&)
        bint empty()
        void pop()
        void push(T&)
        size_t size()
        T& top()

# this is the class type that'll get stored in the priority queue
cdef struct ValuePixelType:
    double value  # pixel value
    int i  # pixel i coordinate in the raster
    int j  # pixel j coordinate in the raster


# this type is used to create a priority queue on a time/coordinate type
ctypedef priority_queue[
    ValuePixelType, deque[ValuePixelType], LessPixel] DistPriorityQueueType

# functor for priority queue of pixels
cdef cppclass LessPixel nogil:
    bint get "operator()"(ValuePixelType& lhs, ValuePixelType& rhs):
        if lhs.value < rhs.value:
            return 1
        return 0


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def find_population_reach(
        numpy.ndarray[float, ndim=2] friction_array,
        numpy.ndarray[float, ndim=2] population_array,
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
        tuple:
        (n_visited,
         2D array of population reach of the same size as input arrays).

    """
    cdef int i, j
    cdef numpy.ndarray[float, ndim=2] pop_coverage = numpy.zeros(
        (n_rows, n_cols), dtype=numpy.float32)
    cdef numpy.ndarray[float, ndim=2] norm_pop_coverage = numpy.zeros(
        (n_rows, n_cols), dtype=numpy.float32)
    cdef numpy.ndarray[numpy.npy_bool, ndim=2] visited = numpy.zeros(
        (n_rows, n_cols), dtype=bool)
    cdef numpy.ndarray[float, ndim=2] current_time = numpy.empty(
        (n_rows, n_cols), dtype=numpy.float32)

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
    cdef float frict_n, c_time, n_time, normalized_pop, population_val
    cdef int i_start, j_start, i_n, j_n
    cdef int min_i, min_j, max_i, max_j

    cdef DistPriorityQueueType dist_queue
    cdef ValuePixelType pixel
    cdef int n_visited, any_visited = 0
    with nogil:
        for i_start in range(core_i, core_i+core_size_i):
            for j_start in range(core_j, core_j+core_size_j):
                population_val = population_array[j_start, i_start]
                if population_val <= 0:
                    continue
                pixel.value = 0
                pixel.i = i_start
                pixel.j = j_start
                dist_queue.push(pixel)
                any_visited = 1
                current_time[j_start, i_start] = 0
                min_i = i_start
                max_i = i_start
                min_j = j_start
                max_j = j_start

                # c_ -- current, n_ -- neighbor
                while dist_queue.size() > 0:
                    pixel = dist_queue.top()
                    dist_queue.pop()
                    c_time = pixel.value
                    i = pixel.i
                    j = pixel.j
                    visited[j, i] = True
                    if i < min_i:
                        min_i = i
                    elif i > max_i:
                        max_i = i
                    if j < min_j:
                        min_j = j
                    elif j > max_j:
                        max_j = j

                    for v in range(8):
                        i_n = i+ioff[v]
                        j_n = j+joff[v]
                        if i_n < 0 or i_n >= n_cols:
                            continue
                        if j_n < 0 or j_n >= n_rows:
                            continue
                        if population_array[j_n, i_n] < 0:
                            # nodata, so skip
                            continue
                        frict_n = friction_array[j_n, i_n]
                        # the nodata value is undefined but will present as 0.
                        if frict_n <= 0:
                            continue
                        n_time = c_time + frict_n*dist_edge[v]
                        if n_time > max_time:
                            continue
                        # if visited before and we got there faster, then skip
                        if visited[j_n, i_n] and n_time >= current_time[j_n, i_n]:
                            continue
                        current_time[j_n, i_n] = n_time
                        if n_time < current_time[j_n, i_n]:
                            raise ValueError(
                                f'just set {n_time} to {current_time[j_n, i_n]} but it did not go less')
                        pixel.value = n_time
                        pixel.i = i_n
                        pixel.j = j_n
                        dist_queue.push(pixel)
                n_visited = 0
                for i in range(min_i, max_i+1):
                    for j in range(min_j, max_j+1):
                        if visited[j, i]:
                            n_visited += 1
                            pop_coverage[j, i] += population_val
                normalized_pop = population_val / float(n_visited)
                for i in range(min_i, max_i+1):
                    for j in range(min_j, max_j+1):
                        if visited[j, i]:
                            norm_pop_coverage[j, i] += normalized_pop
                            # reset for next iteration
                            visited[j, i] = 0
    return any_visited, pop_coverage, norm_pop_coverage
