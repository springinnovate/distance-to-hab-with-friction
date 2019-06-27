"""Distance to habitat with a friction layer."""
import os
import logging
import sys

import ecoshard
import taskgraph

RASTER_ECOSHARD_URL_MAP = {
    'copernicus_hab': 'https://storage.googleapis.com/ecoshard-root/working-shards/masked_nathab_copernicus_md5_420bad770184ce40f028c9c9e02ace4c.tif',
    'esa_hab': 'https://storage.googleapis.com/ecoshard-root/working-shards/masked_nathab_esa_md5_40577bae3ef60519b1043bb8582a07af.tif',
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
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
    hab_fetch_path_map = {}
    # download hab mask and ppl fed equivalent raster
    for raster_id, raster_url in RASTER_ECOSHARD_URL_MAP.items():
        raster_path = os.path.join(ECOSHARD_DIR, os.path.basename(raster_url))
        _ = task_graph.add_task(
            func=ecoshard.download_url,
            args=(raster_url, raster_path),
            target_path_list=[raster_path],
            task_name='fetch %s' % raster_url)
        hab_fetch_path_map[raster_id] = raster_path
    task_graph.join()


if __name__ == '__main__':
    main()
