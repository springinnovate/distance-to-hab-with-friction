"""Distance to habitat with a friction layer."""
import argparse
import logging
import math
import multiprocessing
import os
import queue
import sys
import threading
import time

import ecoshard.geoprocessing
import numpy
from osgeo import gdal
import ecoshard
import taskgraph

import shortest_distances

gdal.SetCacheMax(2**27)

RASTER_ECOSHARD_URL_MAP = {
    # minutes/meter
    'friction_surface': (
        'https://storage.googleapis.com/ecoshard-root/'
        'critical_natural_capital/friction_surface_2015_v1.0-002_'
        'md5_166d17746f5dd49cfb2653d721c2267c.tif'),
    'world_borders': (
        'https://storage.googleapis.com/ecoshard-root/'
        'critical_natural_capital/TM_WORLD_BORDERS-0.3_simplified_'
        'md5_47f2059be8d4016072aa6abe77762021.gpkg'),
}

WORKSPACE_DIR = 'workspace_dist_to_hab_with_friction'
COUNTRY_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'country_workspaces')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
TARGET_NODATA = -1

# max travel distance to cutoff simulation
MAX_TRAVEL_DISTANCE = 9999999
# used to avoid computing paths where the population is too low
POPULATION_COUNT_CUTOFF = 0
# local distance pixel size
TARGET_CELL_LENGTH_M = 2000
TARGET_CELL_AREA_M2 = TARGET_CELL_LENGTH_M**2
# maximum window size to process one set of travel times over
CORE_SIZE = 256 // 2**3

MASK_KEY = 'mask'

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.FileHandler('log.txt'))
logging.getLogger('taskgraph').setLevel(logging.WARN)

SKIP_THESE_COUNTRIES = [
    'Anguilla',
    'Antarctica',
    'Bermuda',
    'French Southern and Antarctic Lands',
    'Gibraltar',
    'Guernsey',
    'Kiribati',
    'Montserrat',
    'Pitcairn Islands',
    'Rwanda',
    'Senegal',
    'Saint Barthelemy',
    'Saint Martin',
    'Solomon Islands',
    'United States Minor Outlying Islands',
    ]

WORLD_ECKERT_IV_WKT = """PROJCRS["unknown",
    BASEGEOGCRS["GCS_unknown",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]],
            ID["EPSG",6326]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["unnamed",
        METHOD["Eckert IV"],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1,
                ID["EPSG",9001]]]]"""


def get_min_nonzero_raster_value(raster_path):
    """Return minimum non-nodata value in raster."""
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    array = band.ReadAsArray()
    if nodata is not None:
        nodata_mask = numpy.isclose(array, nodata)
    else:
        nodata_mask = numpy.zeros(array.shape, dtype=bool)
    valid_mask = ~nodata_mask & numpy.isfinite(array) & (array > 0)
    if valid_mask.any():
        min_value = numpy.min(array[valid_mask])
    else:
        # this happens on tiny areas like vatican, just set it nonsensical
        # and that way it will stand out if it's ever an issue
        min_value = 1
    band = None
    raster = None
    return min_value


def status_monitor(
        start_complete_queue, status_id):
    """Monitor how many steps have been completed and report logging.

    Args:
        start_compelte_queue (queue): first payload is number of updates
            to expect. Subsequent payloads indicate a completed step.
            When totally complete this worker exits.


    Return:
        ``None``
    """
    try:
        n_steps = start_complete_queue.get()
        LOGGER.info(f'status monitor expecting {n_steps} steps')
        steps_complete = 0
        start_time = time.time()
        while True:
            time.sleep(5)
            while True:
                try:
                    _ = start_complete_queue.get_nowait()
                    steps_complete += 1
                except queue.Empty:
                    break
            LOGGER.info(
                f'{status_id} is {steps_complete/n_steps*100:.2f}% complete '
                f'{time.time()-start_time:.1f}s so far')
            if steps_complete == n_steps:
                LOGGER.info(f'done monitoring {status_id}')
                return
    except Exception:
        LOGGER.exception(
            f'something bad happened on status_monitor {status_id}')


def mask_access(
        country_id, friction_raster_path, mask_raster_path,
        max_travel_time, target_mask_access_path):
    """Construct a mask access raster showing where the mask can be reached.

    The mask access raster will have a 1 if it can be reached from that
    given distance travel time.

    Parameters:
        country_id (str): country id just for logging
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        mask_raster_path (str): path to a 0/1 mask indicating the area of
            interest to track travel time to
        max_travel_time (float): maximum time to allow to travel in mins
        target_mask_access_path (str): raster created that
            will contain the count of population that can reach any given
            pixel within the travel time and travel distance constraints.

    Returns:
        None.

    """
    try:
        min_friction = get_min_nonzero_raster_value(friction_raster_path)
        max_travel_distance_in_pixels = math.ceil(
            1/min_friction*max_travel_time/TARGET_CELL_LENGTH_M)

        LOGGER.debug(
            f'min_friction: {min_friction}\n'
            f'max_travel_time: {max_travel_time}\n'
            f'max_travel_distance_in_pixels {max_travel_distance_in_pixels}')

        ecoshard.geoprocessing.new_raster_from_base(
            mask_raster_path, target_mask_access_path,
            gdal.GDT_Byte, [2])

        friction_raster_info = ecoshard.geoprocessing.get_raster_info(
            friction_raster_path)
        raster_x_size, raster_y_size = friction_raster_info['raster_size']

        start_complete_queue = queue.Queue()
        status_monitor_thread = threading.Thread(
            target=status_monitor,
            args=(start_complete_queue, country_id))
        status_monitor_thread.start()

        shortest_distances_worker_thread_list = []
        work_queue = queue.Queue()

        result_queue = queue.Queue()
        for _ in range(1): #multiprocessing.cpu_count()):
            shortest_distances_worker_thread = threading.Thread(
                target=shortest_distances_worker,
                args=(
                    work_queue, result_queue, start_complete_queue,
                    friction_raster_path,
                    mask_raster_path, max_travel_time))
            shortest_distances_worker_thread.start()
            shortest_distances_worker_thread_list.append(
                shortest_distances_worker_thread)

        access_raster_stitcher_thread = threading.Thread(
            target=access_raster_stitcher,
            args=(
                result_queue, start_complete_queue,
                target_mask_access_path))
        access_raster_stitcher_thread.start()

        n_window_x = math.ceil(raster_x_size / CORE_SIZE)
        n_window_y = math.ceil(raster_y_size / CORE_SIZE)
        n_windows = n_window_x * n_window_y
        start_complete_queue.put(n_windows)
        for window_i in range(n_window_x):
            i_core = window_i * CORE_SIZE
            i_offset = i_core - max_travel_distance_in_pixels
            i_size = CORE_SIZE + 2*max_travel_distance_in_pixels
            i_core_size = CORE_SIZE
            if i_offset < 0:
                # shrink the size by the left margin and clamp to 0
                i_size += i_offset
                i_offset = 0

            if i_core+i_core_size >= raster_x_size:
                i_core_size -= i_core+i_core_size - raster_x_size + 1
            if i_offset+i_size >= raster_x_size:
                i_size -= i_offset+i_size - raster_x_size + 1

            for window_j in range(n_window_y):
                j_core = window_j * CORE_SIZE
                j_offset = (
                    j_core - max_travel_distance_in_pixels)
                j_size = CORE_SIZE + 2*max_travel_distance_in_pixels
                j_core_size = CORE_SIZE
                if j_offset < 0:
                    # shrink the size by the left margin and clamp to 0
                    j_size += j_offset
                    j_offset = 0

                if j_core+j_core_size >= raster_y_size:
                    j_core_size -= j_core+j_core_size - raster_y_size + 1
                if j_offset+j_size >= raster_y_size:
                    j_size -= j_offset+j_size - raster_y_size + 1

                work_queue.put(
                    (i_offset, j_offset, i_size, j_size, i_core, j_core,
                     i_core_size, j_core_size))

        work_queue.put(None)
        for worker_thread in shortest_distances_worker_thread_list:
            worker_thread.join()
        LOGGER.info('done with workers')
        result_queue.put(None)
        access_raster_stitcher_thread.join()
        LOGGER.info(
            f'done with access raster worker {target_mask_access_path}')
    except Exception:
        LOGGER.exception(
            f'something bad happened on people_access for '
            f'{target_mask_access_path}')


def shortest_distances_worker(
        work_queue, result_queue, start_complete_queue, friction_raster_path,
        mask_raster_path, max_travel_time):
    """Process shortest distances worker.

    Args:
        work_queue (queue):
        result_queue (queue): the result of a given shortest distance call
            wil be put here as a tuple of
            (n_valid, i_offset, j_offset, people_access)
        start_complete_queue (queue): put a 1 in here for each block complete
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        mask_raster_path (str): path to a per-pixel population
            count raster.
        max_travel_time (float): max travel time in minutes

    Return:
        ``None``
    """
    try:
        friction_raster = gdal.OpenEx(friction_raster_path, gdal.OF_RASTER)
        friction_band = friction_raster.GetRasterBand(1)
        mask_raster = gdal.OpenEx(mask_raster_path, gdal.OF_RASTER)
        mask_band = mask_raster.GetRasterBand(1)
        mask_nodata = mask_band.GetNoDataValue()

        friction_raster_info = ecoshard.geoprocessing.get_raster_info(
            friction_raster_path)
        cell_length = friction_raster_info['pixel_size'][0]
        raster_x_size, raster_y_size = friction_raster_info['raster_size']

        while True:
            payload = work_queue.get()
            if payload is None:
                work_queue.put(None)
                break
            (i_offset, j_offset, i_size, j_size, i_core, j_core,
             i_core_size, j_core_size) = payload
            friction_array = friction_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            mask_array = mask_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            mask_nodata_mask = numpy.zeros(mask_array.shape, dtype=bool) #numpy.isclose(mask_array, mask_nodata)
            # convert it to a byte and drop all the nodata info
            mask_array = (mask_array == 1).astype(numpy.int8)
            #mask_array[mask_nodata_mask] = 0.0

            # doing i_core-i_offset and j_core-j_offset because those
            # do the offsets of the relative size of the array, not the
            # global extents
            mask_reach, mask_count = (
                shortest_distances.find_mask_reach(
                    friction_array, mask_array,
                    cell_length,
                    i_core-i_offset, j_core-j_offset,
                    i_core_size, j_core_size,
                    friction_array.shape[1],
                    friction_array.shape[0],
                    max_travel_time))
            #LOGGER.debug(f'{mask_count} elements seen but \t{numpy.count_nonzero(mask_reach)} elements are in result {i_offset}, {j_offset}')
            result_queue.put(
                (i_offset, j_offset, mask_reach, mask_nodata_mask))
    except Exception:
        LOGGER.exception(
            f'something bad happened on shortest_distances_worker '
            f'{friction_raster_path}')


def access_raster_stitcher(
        work_queue, start_complete_queue,
        target_mask_access_path):
    """Write arrays from the work queue as they come in.

    Args:
        work_queue (queue): expect either None to stop or tuples of
            (n_valid, i_offset, j_offset, people_access,
             normalized_people_access)
        start_complete_queue (queue): put a 1 in here for each block complete
        target_mask_access_path (str): raster created that
            will contain a 1 for any area that can reach the mask in a given
            time period.

    Return:
        ``None``
    """
    try:
        mask_access_raster = gdal.OpenEx(
            target_mask_access_path, gdal.OF_RASTER | gdal.GA_Update)
        mask_access_band = mask_access_raster.GetRasterBand(1)
        mask_access_nodata = mask_access_band.GetNoDataValue()
        while True:
            if (mask_access_band.ReadAsArray()==1).any():
                LOGGER.info(f'PRE got some ones in the band')
            else:
                LOGGER.info('PRE there are NO 1s in mask access band')
            payload = work_queue.get()
            if payload is None:
                LOGGER.info(
                    f'all dones stitching {target_mask_access_path}')

                if (mask_access_band.ReadAsArray()==1).any():
                    LOGGER.info(f'got some ones in the band')
                else:
                    LOGGER.info('there are NO 1s in mask access band')
                break

            (i_offset, j_offset, mask_reach, mask_nodata_mask) = payload
            j_size, i_size = mask_reach.shape
            current_mask_reach = mask_access_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            current_mask_reach[mask_reach == 1] = 1
            #current_mask_reach[
            #    (current_mask_reach == mask_access_nodata) & (~mask_nodata_mask)] = 0
            #current_mask_reach[~mask_nodata_mask] = \
            #    mask_reach[~mask_nodata_mask]
            mask_access_band.WriteArray(
                current_mask_reach, xoff=i_offset, yoff=j_offset)
            LOGGER.debug(mask_reach.size)
            #test_array = mask_access_band.ReadAsArray(
            #    xoff=i_offset, yoff=j_offset,
            #    win_xsize=i_size, win_ysize=j_size)

            #LOGGER.debug(f'stitching {i_offset} {j_offset} with {numpy.count_nonzero(current_mask_reach)} 1s to {target_mask_access_path}')

            start_complete_queue.put(1)
            time.sleep(0.001)

        if (mask_access_band.ReadAsArray()==1).any():
            LOGGER.info(f'got some ones in the band')
        else:
            LOGGER.info('there are NO 1s in mask access band')

        LOGGER.info(
            f'set access rasters to none for {target_mask_access_path}')
        mask_access_band = None
        mask_access_raster = None

    except Exception:
        LOGGER.exception('something bad happened on access_raster_stitcher')
        raise
    LOGGER.info(f'clean exit on {target_mask_access_path}')


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Mask Travel Coverage')
    parser.add_argument(
        '--mask', type=str, required=True,
        help='ecoshard url to mask')
    parser.add_argument(
        '--max_travel_time', required=True,
        type=float, help='travel time in minutes')
    parser.add_argument(
        '--countries', type=str, nargs='+',
        help='comma separated list of countries to simulate')

    args = parser.parse_args()

    max_travel_time = args.max_travel_time

    for dir_path in [WORKSPACE_DIR, CHURN_DIR, ECOSHARD_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    task_graph = taskgraph.TaskGraph(
        CHURN_DIR, multiprocessing.cpu_count()//4, 5.0)
    ecoshard_path_map = {}

    RASTER_ECOSHARD_URL_MAP.update({MASK_KEY: args.mask})
    LOGGER.debug(RASTER_ECOSHARD_URL_MAP)

    for ecoshard_id, ecoshard_url in RASTER_ECOSHARD_URL_MAP.items():
        ecoshard_path = os.path.join(
            ECOSHARD_DIR, os.path.basename(ecoshard_url))
        _ = task_graph.add_task(
            func=ecoshard.download_url,
            args=(ecoshard_url, ecoshard_path),
            target_path_list=[ecoshard_path],
            task_name=f'fetch {ecoshard_url}')
        ecoshard_path_map[ecoshard_id] = ecoshard_path
    task_graph.join()

    mask_id = os.path.basename(os.path.splitext(args.mask)[0])

    world_borders_vector = gdal.OpenEx(
        ecoshard_path_map['world_borders'], gdal.OF_VECTOR)
    world_borders_layer = world_borders_vector.GetLayer()

    area_fid_list = []

    for country_feature in world_borders_layer:
        country_name = country_feature.GetField('NAME')
        if country_name in SKIP_THESE_COUNTRIES:
            LOGGER.debug('skipping %s', country_name)
            continue
        country_geom = country_feature.GetGeometryRef()

        LOGGER.debug(country_name)
        country_geom = country_feature.GetGeometryRef()

        area_fid_list.append((
            country_geom.GetArea(), WORLD_ECKERT_IV_WKT,
            country_name, country_feature.GetFID()))

    world_borders_layer.ResetReading()

    allowed_country_set = None
    if args.countries is not None:
        allowed_country_set = set(
            [name.lower() for name in args.countries])
    mask_access_path_list = []
    for country_index, (
            country_area, target_wkt, country_name, country_fid) in enumerate(
                sorted(area_fid_list, reverse=True)):
        # put the index on there so we can see which one is done first
        if args.countries is not None and (
                country_name.lower() not in allowed_country_set):
            continue
        country_workspace = os.path.join(
            COUNTRY_WORKSPACE_DIR, f'{country_index}_{country_name}')
        os.makedirs(country_workspace, exist_ok=True)
        base_raster_path_list = [
            ecoshard_path_map['friction_surface'],
            ecoshard_path_map[MASK_KEY],
        ]

        # swizzle so it's xmin, ymin, xmax, ymax
        country_feature = world_borders_layer.GetFeature(country_fid)
        LOGGER.debug(f'country name: {country_feature.GetField("NAME")}')
        country_geometry = country_feature.GetGeometryRef()
        country_bb = [
            country_geometry.GetEnvelope()[i] for i in [0, 2, 1, 3]]

        # check the bounding coordinates snap to pixel grid in global coords
        LOGGER.debug(f'lat/lng country_bb: {country_bb}')
        target_bounding_box = ecoshard.geoprocessing.transform_bounding_box(
            country_bb, world_borders_layer.GetSpatialRef().ExportToWkt(),
            target_wkt, edge_samples=11)
        # make sure the bounding coordinates snap to pixel grid
        target_bounding_box[0] -= target_bounding_box[0] % TARGET_CELL_LENGTH_M
        target_bounding_box[1] -= target_bounding_box[1] % TARGET_CELL_LENGTH_M
        target_bounding_box[2] += target_bounding_box[2] % TARGET_CELL_LENGTH_M
        target_bounding_box[3] += target_bounding_box[3] % TARGET_CELL_LENGTH_M
        LOGGER.debug(f'projected country_bb: {target_bounding_box}')

        sinusoidal_friction_path = os.path.join(
            country_workspace, f'{country_name}_friction.tif')
        sinusoidal_mask_path = os.path.join(
            country_workspace, f'{country_name}_{mask_id}.tif')
        sinusoidal_raster_path_list = [
            sinusoidal_friction_path,
            sinusoidal_mask_path,
            ]

        align_precalcualted = all(
            [os.path.exists(path) for path in sinusoidal_raster_path_list])
        projection_task_list = []
        if not align_precalcualted:
            projection_task = task_graph.add_task(
                func=ecoshard.geoprocessing.align_and_resize_raster_stack,
                args=(
                    base_raster_path_list, sinusoidal_raster_path_list,
                    ['average', 'mode'],
                    (TARGET_CELL_LENGTH_M, -TARGET_CELL_LENGTH_M),
                    target_bounding_box),
                kwargs={
                    'target_projection_wkt': WORLD_ECKERT_IV_WKT,
                    'vector_mask_options': {
                        'mask_vector_path': ecoshard_path_map['world_borders'],
                        'mask_vector_where_filter': f'"fid"={country_fid}'
                    }
                },
                ignore_path_list=[ecoshard_path_map['world_borders']],
                target_path_list=sinusoidal_raster_path_list,
                task_name=f'project and clip rasters for {country_name}')
            projection_task_list.append(projection_task)

        mask_travel_coverage_path = os.path.join(
            country_workspace,
            f'mask_coverage_{country_name}_{mask_id}_'
            f'{max_travel_time}m.tif')

        mask_access_task = task_graph.add_task(
            func=mask_access,
            args=(
                country_name,
                sinusoidal_friction_path, sinusoidal_mask_path,
                max_travel_time,
                mask_travel_coverage_path),
            target_path_list=[mask_travel_coverage_path],
            dependent_task_list=projection_task_list,
            task_name='calculating people access for %s' % country_name)
        mask_access_task.is_precalculated()
        LOGGER.debug(f'MASK ACCESS::: {mask_access_task}')

        mask_access_path_list.append((mask_travel_coverage_path, 1))

    LOGGER.debug('create target global mask access layers')
    # warp mask layer to target projection
    warped_mask_access_raster_path = os.path.join(
        WORKSPACE_DIR,
        f"warped_{os.path.basename(ecoshard_path_map[MASK_KEY])}")
    _ = task_graph.add_task(
        func=ecoshard.geoprocessing.warp_raster,
        args=(
            ecoshard_path_map[MASK_KEY],
            (TARGET_CELL_LENGTH_M, -TARGET_CELL_LENGTH_M),
            warped_mask_access_raster_path, 'mode'),
        kwargs={
            'target_projection_wkt': WORLD_ECKERT_IV_WKT,
            'target_bb': [
                -16921202.923, -8460601.461, 16921797.077, 8461398.539],
            'working_dir': WORKSPACE_DIR},
        target_path_list=[warped_mask_access_raster_path],
        task_name=f'warp {warped_mask_access_raster_path}')
    task_graph.close()
    task_graph.join()

    # create mask access
    target_mask_global_access_path = os.path.join(
        WORKSPACE_DIR, f'global_mask_access_{mask_id}_{max_travel_time}m.tif')
    ecoshard.geoprocessing.new_raster_from_base(
        warped_mask_access_raster_path, target_mask_global_access_path,
        gdal.GDT_Byte, [2])

    ecoshard.geoprocessing.stitch_rasters(
        mask_access_path_list,
        ['average']*len(mask_access_path_list),
        (target_mask_global_access_path, 1),
        overlap_algorithm='etch')
    mask_global_access_raster = gdal.OpenEx(
        target_mask_global_access_path, gdal.OF_RASTER | gdal.GA_Update)
    mask_global_access_band = mask_global_access_raster.GetRasterBand(1)
    mask_global_access_band.ComputeStatistics(0)
    mask_global_access_band = None


if __name__ == '__main__':
    main()
