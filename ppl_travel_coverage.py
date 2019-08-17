"""Distance to habitat with a friction layer."""
import time
import os
import logging
import sys

from osgeo import gdal
import ecoshard
import taskgraph
import scipy.sparse.csgraph

import shortest_distances

RASTER_ECOSHARD_URL_MAP = {
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
    'population_layer': r'https://storage.googleapis.com/ecoshard-root/lspop2017_md5_86d653478c1d99d4c6e271bad280637d.tif',
    'world_borders': r'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/TM_WORLD_BORDERS-0.3_simplified_md5_47f2059be8d4016072aa6abe77762021.gpkg'
}

WORKSPACE_DIR = 'workspace_dist_to_hab_with_friction'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
TARGET_NODATA = -1

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, ECOSHARD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
    task_graph = taskgraph.TaskGraph(CHURN_DIR, -1, 5.0)
    ecoshard_path_map = {}
    # download hab mask and ppl fed equivalent raster
    for data_id, data_url in RASTER_ECOSHARD_URL_MAP.items():
        raster_path = os.path.join(ECOSHARD_DIR, os.path.basename(data_url))
        _ = task_graph.add_task(
            func=ecoshard.download_url,
            args=(data_url, raster_path),
            target_path_list=[raster_path],
            task_name='fetch %s' % data_url)
        ecoshard_path_map[data_id] = raster_path
    task_graph.join()

    world_borders_vector = gdal.OpenEx(
        ecoshard_path_map['world_borders'], gdal.OF_VECTOR)
    world_borders_layer = world_borders_vector.GetLayer()

    world_borders_layer.SetAttributeFilter("NAME = 'Bhutan'")

    for feature in world_borders_layer:
        LOGGER.debug(feature.GetField('NAME'))
        continue
        # extract out that country layer and reproject to a UTM zone.
        n_size = 200
        shortest_distances.find_shortest_distances(
            (ecoshard_path_map['friction_surface'], 1),
            10000, 10000, n_size, n_size)
    task_graph.close()


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
    raster = gdal.OpenEx(raster_path_band[0])
    band = raster.GetRasterBand(raster_path_band[1])
    print('opening %s' % str(raster_path_band))
    array = band.ReadAsArray(
        xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)
    print(win_ysize)
    dist_matrix = scipy.sparse.csc_matrix(
        (([1], ([0], [1]))), (array.size, array.size))
    print('making distances')
    start_time = time.time()
    distances = scipy.sparse.csgraph.shortest_path(
        dist_matrix, method='auto', directed=False)
    print(distances)
    print('total time: %s', time.time() - start_time)


if __name__ == '__main__':
    main()
