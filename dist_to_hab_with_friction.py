"""Distance to habitat with a friction layer."""
import time
import os
import logging
import sys

from osgeo import gdal
import ecoshard
import taskgraph
import scipy.sparse.csgraph

RASTER_ECOSHARD_URL_MAP = {
#    'copernicus_hab': 'https://storage.googleapis.com/ecoshard-root/working-shards/masked_nathab_copernicus_md5_420bad770184ce40f028c9c9e02ace4c.tif',
#    'esa_hab': 'https://storage.googleapis.com/ecoshard-root/working-shards/masked_nathab_esa_md5_40577bae3ef60519b1043bb8582a07af.tif',
    # friction layer units: minutes/meter, min=0.0005, max=87.3075
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
    #'population_layer': r'https://storage.googleapis.com/ecoshard-root/lspop2017_md5_86d653478c1d99d4c6e271bad280637d.tif'
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
    for raster_id, raster_url in RASTER_ECOSHARD_URL_MAP.items():
        raster_path = os.path.join(ECOSHARD_DIR, os.path.basename(raster_url))
        # _ = task_graph.add_task(
        #     func=ecoshard.download_url,
        #     args=(raster_url, raster_path),
        #     target_path_list=[raster_path],
        #     task_name='fetch %s' % raster_url)
        ecoshard_path_map[raster_id] = raster_path
    for i in range(4, 10):
        find_shortest_distances(
            (ecoshard_path_map['friction_surface'], 1),
            10000, 10000, 2**i, 2**i)
    task_graph.close()
    task_graph.join()


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
        (array.size, array.size))
    print('making distances')
    start_time = time.time()
    distances = scipy.sparse.csgraph.floyd_warshall(
        dist_matrix, directed=False)
    print(distances)
    print('total time: %s', time.time() - start_time)


if __name__ == '__main__':
    main()
