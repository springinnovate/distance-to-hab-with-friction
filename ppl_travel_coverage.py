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
from ecoshard.geoprocessing.geoprocessing import _create_latitude_m2_area_column
import numpy
from osgeo import gdal
import ecoshard
from ecoshard import taskgraph

import shortest_distances

gdal.SetCacheMax(2**27)

RASTER_ECOSHARD_URL_MAP = {
    # minutes/meter
    #2015 -- 'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/ci_global_restoration/friction_surface_2019_compressed_md5_1be7dd230178a5d395529be7a5e3fb0a.tif',  # 2019
    'population_2017': 'https://storage.googleapis.com/ecoshard-root/population/lspop2017_md5_2e8da6824e4d67f8ea321ba4b585a3a5.tif',
    'lspop_2017_URCA_rural': 'https://storage.googleapis.com/ecoshard-root/population/lspop_2017_URCA_rural_md5_fe4ca0be95aa87a5e0ee6d7db83c0935.tif',
    'lspop_2017_URCA_urban': 'https://storage.googleapis.com/ecoshard-root/population/lspop_2017_URCA_urban_md5_ca344c73dfb71902ab99b4dca2a4a9fc.tif',
    'population_2019': 'https://storage.googleapis.com/ecoshard-root/population/lspop2019_compressed_md5_d0bf03bd0a2378196327bbe6e898b70c.tif',
    #'habitat_mask': 'https://storage.googleapis.com/critical-natural-capital-ecoshards/habmasks/masked_all_nathab_wstreams_esa2015_nodata_md5_ee40334f1fc77bb1804d462c07261c86.tif',
    'world_borders': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/TM_WORLD_BORDERS-0.3_simplified_md5_47f2059be8d4016072aa6abe77762021.gpkg',
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

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.FileHandler('log.txt'))
logging.getLogger('taskgraph').setLevel(logging.DEBUG)

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


def create_population_density(
        wgs84_base_population_count_raster_path,
        target_population_density_raster_path):
    """Convert a wgs84 pop count raster into a population density raster.

    Args:
        wgs84_base_population_count_raster_path (str): path to population
            count raster.
        target_population_density_raster_path (str): path to target
            population.

    Return:
        ``None``
    """
    base_raster_info = ecoshard.geoprocessing.get_raster_info(
        wgs84_base_population_count_raster_path)
    _, lat_min, _, lat_max = base_raster_info['bounding_box']
    n_rows = base_raster_info['raster_size'][1]
    m2_area_per_lat = _create_latitude_m2_area_column(
        lat_min, lat_max, n_rows)

    base_nodata = base_raster_info['nodata'][0]
    target_nodata = -1

    def _div_op(base_array, area):
        """Scale non-nodata by scale."""
        result = numpy.empty(base_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        if base_nodata is not None:
            valid_mask = ~numpy.isclose(base_array, base_nodata)
        else:
            valid_mask = numpy.ones(
                base_array.shape, dtype=bool)
        result[valid_mask] = base_array[valid_mask] / area[valid_mask]
        return result

    # multiply the pixels in the resampled raster by the ratio of
    # the pixel area in the wgs84 units divided by the area of the
    # original pixel
    ecoshard.geoprocessing.raster_calculator(
        [(wgs84_base_population_count_raster_path, 1),
         m2_area_per_lat], _div_op,
        target_population_density_raster_path,
        gdal.GDT_Float32, target_nodata)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='People Travel Coverage')
    parser.add_argument(
        '--population_key', required=True,
        help='population ecoshard key to simulate')
    parser.add_argument(
        '--max_travel_time', required=True,
        type=float, help='travel time in minutes')
    parser.add_argument(
        '--countries', type=str, nargs='+',
        help='comma separated list of countries to simulate')
    args = parser.parse_args()

    population_key = args.population_key
    max_travel_time = args.max_travel_time

    for dir_path in [WORKSPACE_DIR, CHURN_DIR, ECOSHARD_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    task_graph = taskgraph.TaskGraph(
        CHURN_DIR, multiprocessing.cpu_count()//4, 5.0)
    ecoshard_path_map = {}

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

    target_population_density_raster_path = os.path.join(
        CHURN_DIR,
        f'density_{os.path.basename(ecoshard_path_map[population_key])}')
    population_density_task = task_graph.add_task(
        func=create_population_density,
        args=(
            ecoshard_path_map[population_key],
            target_population_density_raster_path),
        target_path_list=[target_population_density_raster_path],
        task_name=f'create population density for {population_key}')
    population_density_task.join()
    ecoshard_path_map[population_key] = target_population_density_raster_path

    world_borders_vector = gdal.OpenEx(
        ecoshard_path_map['world_borders'], gdal.OF_VECTOR)
    world_borders_layer = world_borders_vector.GetLayer()

    area_fid_list = []

    world_eckert_iv_wkt = """PROJCRS["unknown",
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

    for country_feature in world_borders_layer:
        country_name = country_feature.GetField('NAME')
        if country_name in SKIP_THESE_COUNTRIES:
            LOGGER.debug('skipping %s', country_name)
            continue
        country_geom = country_feature.GetGeometryRef()

        LOGGER.debug(country_name)
        country_geom = country_feature.GetGeometryRef()

        area_fid_list.append((
            country_geom.GetArea(), world_eckert_iv_wkt,
            country_name, country_feature.GetFID()))

    world_borders_layer.ResetReading()

    allowed_country_set = None
    if args.countries is not None:
        allowed_country_set = set(
            [name.lower() for name in args.countries])
    people_access_path_list = []
    normalized_people_access_path_list = []
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
            ecoshard_path_map[population_key],
        ]

        # swizzle so it's xmin, ymin, xmax, ymax
        country_feature = world_borders_layer.GetFeature(country_fid)
        LOGGER.debug(f'country name: {country_feature.GetField("NAME")}')
        country_geometry = country_feature.GetGeometryRef()
        country_bb = [
            country_geometry.GetEnvelope()[i] for i in [0, 2, 1, 3]]

        # make sure the bounding coordinates snap to pixel grid in global coords
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
        sinusoidal_population_path = os.path.join(
            country_workspace, f'{country_name}_{population_key}_population.tif')
        sinusoidal_raster_path_list = [
            sinusoidal_friction_path,
            sinusoidal_population_path,
            ]

        align_precalcualted = all(
            [os.path.exists(path) for path in sinusoidal_raster_path_list])
        projection_task_list = []
        if not align_precalcualted:
            projection_task = task_graph.add_task(
                func=ecoshard.geoprocessing.align_and_resize_raster_stack,
                args=(
                    base_raster_path_list, sinusoidal_raster_path_list,
                    ['average', 'average'],
                    (TARGET_CELL_LENGTH_M, -TARGET_CELL_LENGTH_M),
                    target_bounding_box),
                kwargs={
                    'target_projection_wkt': world_eckert_iv_wkt,
                    'vector_mask_options': {
                        'mask_vector_path': ecoshard_path_map['world_borders'],
                        'mask_vector_where_filter': f'"fid"={country_fid}'
                    }
                },
                ignore_path_list=[ecoshard_path_map['world_borders']],
                target_path_list=sinusoidal_raster_path_list,
                task_name=f'project and clip rasters for {country_name}')
            projection_task_list.append(projection_task)

        people_access_path = os.path.join(
            country_workspace,
            f'people_access_{country_name}_{population_key}_'
            f'{max_travel_time}m.tif')
        normalized_people_access_path = os.path.join(
            country_workspace, f'norm_people_access_{country_name}_{population_key}_{max_travel_time}m.tif')

        people_access_task = task_graph.add_task(
            func=people_access,
            args=(
                country_name,
                sinusoidal_friction_path, sinusoidal_population_path,
                max_travel_time,
                people_access_path,
                normalized_people_access_path),
            target_path_list=[
                people_access_path, normalized_people_access_path],
            dependent_task_list=projection_task_list,
            task_name='calculating people access for %s' % country_name)
        people_access_task.is_precalculated()
        LOGGER.debug(f'PEOPLE ACCESS::: {people_access_task}')

        people_access_path_list.append((people_access_path, 1))
        normalized_people_access_path_list.append(
            (normalized_people_access_path, 1))

    LOGGER.debug('create target global population layers')
    # warp population layer to target projection
    warped_pop_raster_path = os.path.join(
        WORKSPACE_DIR, f"warped_{os.path.basename(ecoshard_path_map[population_key])}")
    _ = task_graph.add_task(
        func=ecoshard.geoprocessing.warp_raster,
        args=(
            ecoshard_path_map[population_key],
            (TARGET_CELL_LENGTH_M, -TARGET_CELL_LENGTH_M),
            warped_pop_raster_path, 'average'),
        kwargs={
            'target_projection_wkt': world_eckert_iv_wkt,
            'target_bb': [
                -16921202.923, -8460601.461, 16921797.077, 8461398.539],
            'working_dir': WORKSPACE_DIR},
        target_path_list=[warped_pop_raster_path],
        task_name=f'warp {warped_pop_raster_path}')
    task_graph.close()
    task_graph.join()

    # create access and normalized access paths
    target_people_global_access_path = os.path.join(
        WORKSPACE_DIR, f'global_people_access_{population_key}_{max_travel_time}m.tif')
    ecoshard.geoprocessing.new_raster_from_base(
        warped_pop_raster_path, target_people_global_access_path,
        gdal.GDT_Float32, [-1])
    target_normalized_people_global_access_path = os.path.join(
        WORKSPACE_DIR, f'global_normalized_people_access_{population_key}_{max_travel_time}m.tif')
    ecoshard.geoprocessing.new_raster_from_base(
        warped_pop_raster_path,
        target_normalized_people_global_access_path, gdal.GDT_Float32,
        [-1])

    ecoshard.geoprocessing.stitch_rasters(
        people_access_path_list,
        ['average']*len(people_access_path_list),
        (target_people_global_access_path, 1),
        overlap_algorithm='etch')
    people_global_access_raster = gdal.OpenEx(
        target_people_global_access_path, gdal.OF_RASTER | gdal.GA_Update)
    people_global_access_band = people_global_access_raster.GetRasterBand(1)
    people_global_access_band.ComputeStatistics(0)
    people_global_access_band = None
    ecoshard.geoprocessing.stitch_rasters(
        normalized_people_access_path_list,
        ['average']*len(normalized_people_access_path_list),
        (target_normalized_people_global_access_path, 1),
        overlap_algorithm='etch')
    normalized_people_global_access_raster = gdal.OpenEx(
        target_normalized_people_global_access_path, gdal.OF_RASTER | gdal.GA_Update)
    normalized_people_global_access_band = normalized_people_global_access_raster.GetRasterBand(1)
    normalized_people_global_access_band.ComputeStatistics(0)
    normalized_people_global_access_band = None


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


def people_access(
        country_id, friction_raster_path, population_density_raster_path,
        max_travel_time, target_people_access_path,
        target_normalized_people_access_path):
    """Construct a people access raster showing where people can reach.

    The people access raster will have a value of population count per pixel
    which can reach that pixel within a cutoff of `max_travel_time` or
    `max_travel_distance`.

    Parameters:
        country_id (str): country id just for logging
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        population_density_raster_path (str): path to a per-pixel population
            density pop/m^2 raster.
        max_travel_time (float): maximum time to allow to travel in mins
        target_people_access_path (str): raster created that
            will contain the count of population that can reach any given
            pixel within the travel time and travel distance constraints.
        target_normalized_people_access_path (str): raster created
            that contains a normalized count of population that can
            reach any pixel within the travel time. Population is normalized
            by dividing the source population by the number of pixels that
            it can reach such that the sum of the entire reachable area
            equals the original population count. Useful for aggregating of
            "number of people that can reach this area" and similar
            calculations.

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
            population_density_raster_path, target_people_access_path,
            gdal.GDT_Float32, [-1])
        ecoshard.geoprocessing.new_raster_from_base(
            population_density_raster_path, target_normalized_people_access_path,
            gdal.GDT_Float32, [-1])

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
        for _ in range(multiprocessing.cpu_count()):
            shortest_distances_worker_thread = threading.Thread(
                target=shortest_distances_worker,
                args=(
                    work_queue, result_queue, start_complete_queue,
                    friction_raster_path,
                    population_density_raster_path, max_travel_time))
            shortest_distances_worker_thread.start()
            shortest_distances_worker_thread_list.append(
                shortest_distances_worker_thread)

        access_raster_stitcher_thread = threading.Thread(
            target=access_raster_stitcher,
            args=(
                result_queue, start_complete_queue,
                #habitat_raster_path,
                target_people_access_path,
                target_normalized_people_access_path))
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
        LOGGER.info(f'done with workers')
        result_queue.put(None)
        access_raster_stitcher_thread.join()
        LOGGER.info(f'done with access raster worker {target_people_access_path}')
    except Exception:
        LOGGER.exception(f'something bad happened on people_access for {target_people_access_path}')


def shortest_distances_worker(
        work_queue, result_queue, start_complete_queue, friction_raster_path,
        population_density_raster_path, max_travel_time):
    """Process shortest distances worker.

    Args:
        work_queue (queue):
        result_queue (queue): the result of a given shortest distance call
            wil be put here as a tuple of
            (n_valid, i_offset, j_offset, people_access,
             normalized_people_access)
        start_complete_queue (queue): put a 1 in here for each block complete
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        population_density_raster_path (str): path to a per-pixel population
            count raster.
        max_travel_time (float): max travel time in minutes

    Return:
        ``None``
    """
    try:
        friction_raster = gdal.OpenEx(friction_raster_path, gdal.OF_RASTER)
        friction_band = friction_raster.GetRasterBand(1)
        population_raster = gdal.OpenEx(
            population_density_raster_path, gdal.OF_RASTER)
        population_band = population_raster.GetRasterBand(1)
        population_nodata = population_band.GetNoDataValue()

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
            population_array = population_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            pop_nodata_mask = numpy.isclose(
                population_array, population_nodata)
            population_array[pop_nodata_mask] = 0.0

            # doing i_core-i_offset and j_core-j_offset because those
            # do the offsets of the relative size of the array, not the
            # global extents
            population_reach, norm_population_reach = (
                shortest_distances.find_population_reach(
                    friction_array, population_array,
                    cell_length,
                    i_core-i_offset, j_core-j_offset,
                    i_core_size, j_core_size,
                    friction_array.shape[1],
                    friction_array.shape[0],
                    max_travel_time))
            result_queue.put(
                (i_offset, j_offset, population_reach, norm_population_reach, pop_nodata_mask))
    except Exception:
        LOGGER.exception(f'something bad happened on shortest_distances_worker {friction_raster_path}')


def access_raster_stitcher(
        work_queue, start_complete_queue,
        target_people_access_path,
        target_normalized_people_access_path):
    """Write arrays from the work queue as they come in.

    Args:
        work_queue (queue): expect either None to stop or tuples of
            (n_valid, i_offset, j_offset, people_access,
             normalized_people_access)
        start_complete_queue (queue): put a 1 in here for each block complete
        target_people_access_path (str): raster created that
            will contain the count of population that can reach any given
            pixel within the travel time and travel distance constraints.
        target_normalized_people_access_path  (str): raster created
            that contains a normalized count of population that can
            reach any pixel within the travel time. Population is normalized
            by dividing the source population by the number of pixels that
            it can reach such that the sum of the entire reachable area
            equals the original population count. Useful for aggregating of
            "number of people that can reach this area" and similar
            calculations.

    Return:
        ``None``
    """
    try:
        people_access_raster = gdal.OpenEx(
            target_people_access_path, gdal.OF_RASTER | gdal.GA_Update)
        people_access_band = people_access_raster.GetRasterBand(1)
        normalized_people_access_raster = gdal.OpenEx(
            target_normalized_people_access_path,
            gdal.OF_RASTER | gdal.GA_Update)
        normalized_people_access_band = (
            normalized_people_access_raster.GetRasterBand(1))
        while True:
            payload = work_queue.get()
            if payload is None:
                LOGGER.info(
                    f'all done with {target_people_access_path} and '
                    f'{target_normalized_people_access_path}')
                break

            (i_offset, j_offset, population_reach, norm_population_reach, pop_nodata_mask) = payload
            j_size, i_size = population_reach.shape
            current_pop_reach = people_access_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            current_pop_reach[(current_pop_reach == -1) & (~pop_nodata_mask)] = 0
            current_pop_reach[~pop_nodata_mask] += \
                population_reach[~pop_nodata_mask] * (TARGET_CELL_AREA_M2)
            people_access_band.WriteArray(
                current_pop_reach, xoff=i_offset, yoff=j_offset)

            current_norm_pop_reach = (
                normalized_people_access_band.ReadAsArray(
                    xoff=i_offset, yoff=j_offset,
                    win_xsize=i_size, win_ysize=j_size))
            current_norm_pop_reach[(current_norm_pop_reach == -1) & (~pop_nodata_mask)] = 0
            current_norm_pop_reach[~pop_nodata_mask] += (
                norm_population_reach[~pop_nodata_mask] * TARGET_CELL_AREA_M2)
            normalized_people_access_band.WriteArray(
                current_norm_pop_reach, xoff=i_offset, yoff=j_offset)
            start_complete_queue.put(1)

        LOGGER.info(
            f'set access rasters to none for {target_people_access_path} and '
            f'{target_normalized_people_access_path}')
        people_access_raster = None
        normalized_people_access_raster = None
        people_access_band = None
        normalized_people_access_band = None
        LOGGER.info(
            f'done writing to {target_people_access_path} and '
            f'{target_normalized_people_access_path}')
    except Exception:
        LOGGER.exception(f'something bad happened on access_raster_stitcher')
        raise


if __name__ == '__main__':
    main()
=======
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

import pygeoprocessing
from pygeoprocessing.geoprocessing import _create_latitude_m2_area_column
import numpy
from osgeo import gdal
import ecoshard
import taskgraph

import shortest_distances

gdal.SetCacheMax(2**27)

RASTER_ECOSHARD_URL_MAP = {
    # minutes/meter
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
    'population_2017': 'https://storage.googleapis.com/ecoshard-root/population/lspop2017_md5_2e8da6824e4d67f8ea321ba4b585a3a5.tif',
    'lspop_2017_URCA_rural': 'https://storage.googleapis.com/ecoshard-root/population/lspop_2017_URCA_rural_md5_fe4ca0be95aa87a5e0ee6d7db83c0935.tif',
    'lspop_2017_URCA_urban': 'https://storage.googleapis.com/ecoshard-root/population/lspop_2017_URCA_urban_md5_ca344c73dfb71902ab99b4dca2a4a9fc.tif',
    'habitat_mask': 'https://storage.googleapis.com/critical-natural-capital-ecoshards/habmasks/masked_all_nathab_wstreams_esa2015_md5_c291ff6ef7db1d5ff4d95a82e0f035de.tif',
    'world_borders': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/TM_WORLD_BORDERS-0.3_simplified_md5_47f2059be8d4016072aa6abe77762021.gpkg',
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
CORE_SIZE = 256

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.FileHandler('log.txt'))
logging.getLogger('taskgraph').setLevel(logging.ERROR)

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
    'Saint Barthelemy',
    'Saint Martin',
    'Solomon Islands',
    'United States Minor Outlying Islands',
    ]


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


def create_population_density(
        wgs84_base_population_count_raster_path,
        target_population_density_raster_path):
    """Convert a wgs84 pop count raster into a population density raster.

    Args:
        wgs84_base_population_count_raster_path (str): path to population
            count raster.
        target_population_density_raster_path (str): path to target
            population.

    Return:
        ``None``
    """
    base_raster_info = pygeoprocessing.get_raster_info(
        wgs84_base_population_count_raster_path)
    _, lat_min, _, lat_max = base_raster_info['bounding_box']
    n_rows = base_raster_info['raster_size'][1]
    m2_area_per_lat = _create_latitude_m2_area_column(
        lat_min, lat_max, n_rows)

    base_nodata = base_raster_info['nodata'][0]
    target_nodata = -1

    def _div_op(base_array, area):
        """Scale non-nodata by scale."""
        result = numpy.empty(base_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        if base_nodata is not None:
            valid_mask = ~numpy.isclose(base_array, base_nodata)
        else:
            valid_mask = numpy.ones(
                base_array.shape, dtype=bool)
        result[valid_mask] = base_array[valid_mask] / area[valid_mask]
        return result

    # multiply the pixels in the resampled raster by the ratio of
    # the pixel area in the wgs84 units divided by the area of the
    # original pixel
    pygeoprocessing.raster_calculator(
        [(wgs84_base_population_count_raster_path, 1),
         m2_area_per_lat], _div_op,
        target_population_density_raster_path,
        gdal.GDT_Float32, target_nodata)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='People Travel Coverage')
    parser.add_argument(
        '--population_key', required=True,
        help='population ecoshard key to simulate')
    parser.add_argument(
        '--max_travel_time', required=True,
        type=float, help='travel time in minutes')
    parser.add_argument(
        '--pixel_size_m', required=True,
        type=float, help='pixel size in meters')
    parser.add_argument(
        '--countries', type=str, nargs='+',
        help='comma separated list of countries to simulate')
    args = parser.parse_args()

    population_key = args.population_key
    max_travel_time = args.max_travel_time

    for dir_path in [WORKSPACE_DIR, CHURN_DIR, ECOSHARD_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    task_graph = taskgraph.TaskGraph(CHURN_DIR, multiprocessing.cpu_count())
    ecoshard_path_map = {}

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

    target_population_density_raster_path = os.path.join(
        CHURN_DIR,
        f'density_{os.path.basename(ecoshard_path_map[population_key])}')
    population_density_task = task_graph.add_task(
        func=create_population_density,
        args=(
            ecoshard_path_map[population_key],
            target_population_density_raster_path),
        target_path_list=[target_population_density_raster_path],
        task_name=f'create population density for {population_key}')
    population_density_task.join()
    ecoshard_path_map[population_key] = target_population_density_raster_path

    world_borders_vector = gdal.OpenEx(
        ecoshard_path_map['world_borders'], gdal.OF_VECTOR)
    world_borders_layer = world_borders_vector.GetLayer()

    area_fid_list = []

    world_eckert_iv_wkt = """PROJCRS["unknown",
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

    for country_feature in world_borders_layer:
        country_name = country_feature.GetField('NAME')
        if country_name in SKIP_THESE_COUNTRIES:
            LOGGER.debug('skipping %s', country_name)
            continue
        country_geom = country_feature.GetGeometryRef()

        LOGGER.debug(country_name)
        country_geom = country_feature.GetGeometryRef()

        area_fid_list.append((
            country_geom.GetArea(), world_eckert_iv_wkt,
            country_name, country_feature.GetFID()))

    world_borders_layer.ResetReading()

    allowed_country_set = None
    if args.countries is not None:
        allowed_country_set = set(
            [name.lower() for name in args.countries])
    people_access_path_list = []
    normalized_people_access_path_list = []
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
            ecoshard_path_map[population_key],
            ecoshard_path_map['habitat_mask'],
        ]

        # swizzle so it's xmin, ymin, xmax, ymax
        country_feature = world_borders_layer.GetFeature(country_fid)
        LOGGER.debug(f'country name: {country_feature.GetField("NAME")}')
        country_geometry = country_feature.GetGeometryRef()
        country_bb = [
            country_geometry.GetEnvelope()[i] for i in [0, 2, 1, 3]]

        # make sure the bounding coordinates snap to pixel grid in global coords
        LOGGER.debug(f'lat/lng country_bb: {country_bb}')
        target_bounding_box = pygeoprocessing.transform_bounding_box(
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
        sinusoidal_population_path = os.path.join(
            country_workspace,
            f'{country_name}_population_{population_key}.tif')
        sinusoidal_hab_path = os.path.join(
            country_workspace, f'sinusoidal_{country_name}_hab.tif')
        sinusoidal_raster_path_list = [
            sinusoidal_friction_path, sinusoidal_population_path,
            sinusoidal_hab_path]

        projection_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            args=(
                base_raster_path_list, sinusoidal_raster_path_list,
                ['near']*len(base_raster_path_list),
                (TARGET_CELL_LENGTH_M, -TARGET_CELL_LENGTH_M),
                target_bounding_box),
            kwargs={
                'target_projection_wkt': world_eckert_iv_wkt,
                'vector_mask_options': {
                    'mask_vector_path': ecoshard_path_map['world_borders'],
                    'mask_vector_where_filter': f'"fid"={country_fid}'
                }
            },
            target_path_list=sinusoidal_raster_path_list,
            task_name=f'project and clip rasters for {country_name}')

        people_access_path = os.path.join(
            country_workspace,
            f'people_access_{country_name}_{population_key}_{max_travel_time}m.tif')
        normalized_people_access_path = os.path.join(
            country_workspace, f'norm_people_access_{country_name}_{max_travel_time}m.tif')

        _ = task_graph.add_task(
            func=people_access,
            args=(
                country_name,
                sinusoidal_friction_path, sinusoidal_population_path,
                sinusoidal_hab_path, max_travel_time,
                people_access_path,
                normalized_people_access_path),
            target_path_list=[
                people_access_path, normalized_people_access_path],
            dependent_task_list=[projection_task],
            task_name='calculating people access for %s' % country_name)
        people_access_path_list.append((people_access_path, 1))
        normalized_people_access_path_list.append(
            (normalized_people_access_path, 1))

    LOGGER.debug('create target global population layers')
    # warp population layer to target projection
    warped_pop_raster_path = os.path.join(
        WORKSPACE_DIR, f"warped_{os.path.basename(ecoshard_path_map[population_key])}")
    _ = task_graph.add_task(
        func=pygeoprocessing.warp_raster,
        args=(
            ecoshard_path_map[population_key],
            (TARGET_CELL_LENGTH_M, -TARGET_CELL_LENGTH_M),
            warped_pop_raster_path, 'near'),
        kwargs={
            'target_projection_wkt': world_eckert_iv_wkt,
            'target_bb': [
                -16921202.923, -8460601.461, 16921797.077, 8461398.539],
            'working_dir': WORKSPACE_DIR},
        target_path_list=[warped_pop_raster_path],
        task_name=f'warp {warped_pop_raster_path}')
    task_graph.close()
    task_graph.join()

    # create access and normalized access paths
    target_people_global_access_path = os.path.join(
        WORKSPACE_DIR, f'global_people_access_{population_key}_{max_travel_time}m.tif')
    pygeoprocessing.new_raster_from_base(
        warped_pop_raster_path, target_people_global_access_path,
        gdal.GDT_Float32, [-1])
    target_normalized_people_global_access_path = os.path.join(
        WORKSPACE_DIR, f'global_normalized_people_access_{population_key}_{max_travel_time}m.tif')
    pygeoprocessing.new_raster_from_base(
        warped_pop_raster_path,
        target_normalized_people_global_access_path, gdal.GDT_Float32,
        [-1])

    pygeoprocessing.stitch_rasters(
        people_access_path_list,
        ['near']*len(people_access_path_list),
        (target_people_global_access_path, 1),
        overlap_algorithm='etch')
    people_global_access_raster = gdal.OpenEx(
        target_people_global_access_path, gdal.OF_RASTER | gdal.GA_Update)
    people_global_access_band = people_global_access_raster.GetRasterBand(1)
    people_global_access_band.ComputeStatistics(0)
    people_global_access_band = None
    pygeoprocessing.stitch_rasters(
        normalized_people_access_path_list,
        ['near']*len(normalized_people_access_path_list),
        (target_normalized_people_global_access_path, 1),
        overlap_algorithm='etch')
    normalized_people_global_access_raster = gdal.OpenEx(
        target_normalized_people_global_access_path, gdal.OF_RASTER | gdal.GA_Update)
    normalized_people_global_access_band = normalized_people_global_access_raster.GetRasterBand(1)
    normalized_people_global_access_band.ComputeStatistics(0)
    normalized_people_global_access_band = None


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


def people_access(
        country_id, friction_raster_path, population_density_raster_path,
        habitat_raster_path, max_travel_time, target_people_access_path,
        target_normalized_people_access_path):
    """Construct a people access raster showing where people can reach.

    The people access raster will have a value of population count per pixel
    which can reach that pixel within a cutoff of `max_travel_time` or
    `max_travel_distance`.

    Parameters:
        country_id (str): country id just for logging
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        population_density_raster_path (str): path to a per-pixel population
            density pop/m^2 raster.
        max_travel_time (float): maximum time to allow to travel in mins
        target_people_access_path (str): raster created that
            will contain the count of population that can reach any given
            pixel within the travel time and travel distance constraints.
        target_normalized_people_access_path (str): raster created
            that contains a normalized count of population that can
            reach any pixel within the travel time. Population is normalized
            by dividing the source population by the number of pixels that
            it can reach such that the sum of the entire reachable area
            equals the original population count. Useful for aggregating of
            "number of people that can reach this area" and similar
            calculations.

    Returns:
        None.

    """
    min_friction = get_min_nonzero_raster_value(friction_raster_path)
    max_travel_distance_in_pixels = math.ceil(
        1/min_friction*max_travel_time/TARGET_CELL_LENGTH_M)

    LOGGER.debug(
        f'min_friction: {min_friction}\n'
        f'max_travel_time: {max_travel_time}\n'
        f'max_travel_distance_in_pixels {max_travel_distance_in_pixels}')

    pygeoprocessing.new_raster_from_base(
        population_density_raster_path, target_people_access_path,
        gdal.GDT_Float32, [-1])
    pygeoprocessing.new_raster_from_base(
        population_density_raster_path, target_normalized_people_access_path,
        gdal.GDT_Float32, [-1])

    friction_raster_info = pygeoprocessing.get_raster_info(
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
    for _ in range(multiprocessing.cpu_count()):
        shortest_distances_worker_thread = threading.Thread(
            target=shortest_distances_worker,
            args=(
                work_queue, result_queue, start_complete_queue,
                friction_raster_path,
                population_density_raster_path, max_travel_time))
        shortest_distances_worker_thread.start()
        shortest_distances_worker_thread_list.append(
            shortest_distances_worker_thread)

    access_raster_stitcher_thread = threading.Thread(
        target=access_raster_stitcher,
        args=(
            result_queue, start_complete_queue,
            habitat_raster_path,
            target_people_access_path,
            target_normalized_people_access_path))
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
    LOGGER.info(f'done with workers')
    result_queue.put(None)
    access_raster_stitcher_thread.join()
    LOGGER.info(f'done with access raster worker {target_people_access_path}')


def shortest_distances_worker(
        work_queue, result_queue, start_complete_queue, friction_raster_path,
        population_density_raster_path, max_travel_time):
    """Process shortest distances worker.

    Args:
        work_queue (queue):
        result_queue (queue): the result of a given shortest distance call
            wil be put here as a tuple of
            (n_valid, i_offset, j_offset, people_access,
             normalized_people_access)
        start_complete_queue (queue): put a 1 in here for each block complete
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        population_density_raster_path (str): path to a per-pixel population
            count raster.
        max_travel_time (float): max travel time in minutes

    Return:
        ``None``
    """
    friction_raster = gdal.OpenEx(friction_raster_path, gdal.OF_RASTER)
    friction_band = friction_raster.GetRasterBand(1)
    population_raster = gdal.OpenEx(
        population_density_raster_path, gdal.OF_RASTER)
    population_band = population_raster.GetRasterBand(1)
    population_nodata = population_band.GetNoDataValue()

    friction_raster_info = pygeoprocessing.get_raster_info(
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
        population_array = population_band.ReadAsArray(
            xoff=i_offset, yoff=j_offset,
            win_xsize=i_size, win_ysize=j_size)
        pop_nodata_mask = numpy.isclose(
            population_array, population_nodata)
        total_population = numpy.sum(population_array[~pop_nodata_mask])
        # don't route population where there isn't any
        if total_population < POPULATION_COUNT_CUTOFF:
            LOGGER.debug(
                f'skipping because {total_population} < '
                f'{POPULATION_COUNT_CUTOFF}')
            start_complete_queue.put(1)
            continue

        population_array[pop_nodata_mask] = 0.0

        # doing i_core-i_offset and j_core-j_offset because those
        # do the offsets of the relative size of the array, not the
        # global extents
        n_visited, population_reach, norm_population_reach = (
            shortest_distances.find_population_reach(
                friction_array, population_array,
                cell_length,
                i_core-i_offset, j_core-j_offset,
                i_core_size, j_core_size,
                friction_array.shape[1],
                friction_array.shape[0],
                max_travel_time))
        if n_visited == 0:
            LOGGER.debug(
                f'no need to write an empty array skipping '
                f'{i_offset} {j_offset}')
            start_complete_queue.put(1)
            continue
        result_queue.put(
            (i_offset, j_offset, population_reach, norm_population_reach))


def access_raster_stitcher(
        work_queue, start_complete_queue,
        habitat_raster_path, target_people_access_path,
        target_normalized_people_access_path):
    """Write arrays from the work queue as they come in.

    Args:
        work_queue (queue): expect either None to stop or tuples of
            (n_valid, i_offset, j_offset, people_access,
             normalized_people_access)
        start_complete_queue (queue): put a 1 in here for each block complete
        habitat_raster_path (str): path to habitat layer used to determine
            where to mask by habitat.
        target_people_access_path (str): raster created that
            will contain the count of population that can reach any given
            pixel within the travel time and travel distance constraints.
        target_normalized_people_access_path  (str): raster created
            that contains a normalized count of population that can
            reach any pixel within the travel time. Population is normalized
            by dividing the source population by the number of pixels that
            it can reach such that the sum of the entire reachable area
            equals the original population count. Useful for aggregating of
            "number of people that can reach this area" and similar
            calculations.

    Return:
        ``None``
    """
    try:
        habitat_raster = gdal.OpenEx(
            habitat_raster_path, gdal.OF_RASTER)
        habitat_band = habitat_raster.GetRasterBand(1)
        people_access_raster = gdal.OpenEx(
            target_people_access_path, gdal.OF_RASTER | gdal.GA_Update)
        people_access_band = people_access_raster.GetRasterBand(1)
        normalized_people_access_raster = gdal.OpenEx(
            target_normalized_people_access_path,
            gdal.OF_RASTER | gdal.GA_Update)
        normalized_people_access_band = (
            normalized_people_access_raster.GetRasterBand(1))
        while True:
            payload = work_queue.get()
            if payload is None:
                LOGGER.info(
                    f'all done with {target_people_access_path} and '
                    f'{target_normalized_people_access_path}')
                break

            (i_offset, j_offset, population_reach, norm_population_reach) = payload
            j_size, i_size = population_reach.shape
            LOGGER.info(f'got payload for {i_offset} {j_offset} {i_size} {j_size}')
            current_pop_reach = people_access_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            habitat_array = habitat_band.ReadAsArray(
                xoff=i_offset, yoff=j_offset,
                win_xsize=i_size, win_ysize=j_size)
            valid_mask = habitat_array == 1
            current_pop_reach[(current_pop_reach == -1) & valid_mask] = 0
            current_pop_reach[valid_mask] += population_reach[valid_mask] * (
                TARGET_CELL_AREA_M2)
            people_access_band.WriteArray(
                current_pop_reach, xoff=i_offset, yoff=j_offset)

            current_norm_pop_reach = (
                normalized_people_access_band.ReadAsArray(
                    xoff=i_offset, yoff=j_offset,
                    win_xsize=i_size, win_ysize=j_size))
            current_norm_pop_reach[
                (current_norm_pop_reach == -1) & valid_mask] = 0
            current_norm_pop_reach[valid_mask] += (
                norm_population_reach[valid_mask] * TARGET_CELL_AREA_M2)
            normalized_people_access_band.WriteArray(
                current_norm_pop_reach, xoff=i_offset, yoff=j_offset)
            start_complete_queue.put(1)

        LOGGER.info(
            f'set access rasters to none for {target_people_access_path} and '
            f'{target_normalized_people_access_path}')
        people_access_raster = None
        normalized_people_access_raster = None
        people_access_band = None
        normalized_people_access_band = None
        LOGGER.info(
            f'done writing to {target_people_access_path} and '
            f'{target_normalized_people_access_path}')
    except Exception:
        LOGGER.exception(f'something bad happened on access_raster_stitcher')
        raise


if __name__ == '__main__':
    main()
