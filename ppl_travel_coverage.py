"""Distance to habitat with a friction layer."""
import datetime
import multiprocessing
import time
import os
import logging
import sys

import pygeoprocessing
import numpy
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import ecoshard
import taskgraph
import scipy.sparse.csgraph

import shortest_distances

RASTER_ECOSHARD_URL_MAP = {
    # minutes/meter
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
    'population_layer': r'https://storage.googleapis.com/ecoshard-root/lspop2017_md5_86d653478c1d99d4c6e271bad280637d.tif',
    'world_borders': r'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/TM_WORLD_BORDERS-0.3_simplified_md5_47f2059be8d4016072aa6abe77762021.gpkg',
    'habitat_mask': r'https://storage.googleapis.com/critical-natural-capital-ecoshards/masked_nathab_esa_md5_40577bae3ef60519b1043bb8582a07af.tif'
}

WORKSPACE_DIR = 'workspace_dist_to_hab_with_friction'
COUNTRY_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'country_workspaces')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
TARGET_NODATA = -1
SKIP_THESE_COUNTRIES = ['United States Minor Outlying Islands']

# max travel time in minutes, basing off of half of a travel day (roundtrip)
MAX_TRAVEL_TIME = 1*60  # minutes
# max travel distance to cutoff simulation
MAX_TRAVEL_DISTANCE = 20000
# used to avoid computing paths where the population is too low
POPULATION_COUNT_CUTOFF = 100

TASKGRAPH_WORKERS = -1  # multiprocessing.cpu_count()

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
    task_graph = taskgraph.TaskGraph(CHURN_DIR, TASKGRAPH_WORKERS, 5.0)
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

    area_fid_list = []
    for country_feature in world_borders_layer:
        country_name = country_feature.GetField('NAME')
        if country_name in SKIP_THESE_COUNTRIES:
            continue
        country_geom = country_feature.GetGeometryRef()

        LOGGER.debug(country_name)
        country_geom = country_feature.GetGeometryRef()
        # find EPSG code that would be central to the country
        centroid_geom = country_geom.Centroid()
        utm_code = (numpy.floor((centroid_geom.GetX()+180)/6) % 60)+1
        lat_code = 6 if centroid_geom.GetY() > 0 else 7
        epsg_code = int('32%d%02d' % (lat_code, utm_code))
        LOGGER.debug(epsg_code)
        epsg_srs = osr.SpatialReference()
        epsg_srs.ImportFromEPSG(epsg_code)

        area_fid_list.append((
            country_geom.GetArea(), epsg_srs.ExportToWkt(), country_name,
            country_feature.GetFID()))

    world_borders_layer.ResetReading()

    friction_raster_info = pygeoprocessing.get_raster_info(
        ecoshard_path_map['friction_surface'])
    for country_index, (country_area, epsg_wkt, country_name, country_fid) in enumerate(
            sorted(area_fid_list)):
        # put the index on there so we can see which one is done first
        country_workspace = os.path.join(
            COUNTRY_WORKSPACE_DIR, '%d_%s' % (country_index, country_name))
        try:
            os.makedirs(country_workspace)
        except OSError:
            pass
        country_vector_path = os.path.join(
            country_workspace, '%s.gpkg' % country_name)
        country_vector_complete_token_path = (
            '%s.COMPLETE' % country_vector_path)
        extract_country_task = task_graph.add_task(
            func=extract_and_project_feature,
            priority=-country_index,
            args=(
                ecoshard_path_map['world_borders'], country_fid, epsg_wkt,
                country_vector_path, country_vector_complete_token_path),
            ignore_path_list=[country_vector_path],
            target_path_list=[country_vector_complete_token_path],
            task_name='make local country for %s' % country_name)
        base_raster_path_list = [
            ecoshard_path_map['friction_surface'],
            ecoshard_path_map['population_layer'],
            ecoshard_path_map['habitat_mask'],
        ]
        wgs84_raster_path_list = [
            os.path.join(
                country_workspace, 'wgs84_%s_friction.tif' % country_name),
            os.path.join(
                country_workspace, 'wgs84_%s_population.tif' % country_name),
            os.path.join(
                country_workspace, 'wsg84_%s_hab_mask.tif' % country_name)
        ]
        clip_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            priority=-country_index,
            args=(
                base_raster_path_list, wgs84_raster_path_list,
                ['near']*len(base_raster_path_list),
                friction_raster_info['pixel_size'], 'intersection'),
            kwargs={
                'base_vector_path_list': [country_vector_path],
                'target_sr_wkt': friction_raster_info['projection']
                },
            dependent_task_list=[extract_country_task],
            target_path_list=wgs84_raster_path_list,
            task_name='project for %s' % country_name)
        utm_friction_path = os.path.join(
            country_workspace, 'utm_%s_friction.tif' % country_name)
        utm_population_path = os.path.join(
            country_workspace, 'utm_%s_population.tif' % country_name)
        utm_hab_path = os.path.join(
            country_workspace, 'utm_%s_hab.tif' % country_name)
        utm_raster_path_list = [
            utm_friction_path, utm_population_path, utm_hab_path]

        m_per_deg = length_of_degree(centroid_geom.GetY())
        target_pixel_size = (
            m_per_deg*friction_raster_info['pixel_size'][0],
            m_per_deg*friction_raster_info['pixel_size'][1])
        # this will project values outside to 0 since there's not a nodata
        # value defined
        projection_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
            priority=-country_index,
            args=(
                wgs84_raster_path_list, utm_raster_path_list,
                ['near']*len(base_raster_path_list),
                target_pixel_size, 'intersection'),
            kwargs={
                'base_vector_path_list': [country_vector_path],
                'target_sr_wkt': epsg_srs.ExportToWkt(),
                'vector_mask_options': {
                    'mask_vector_path': country_vector_path,
                },
            },
            dependent_task_list=[clip_task],
            target_path_list=utm_raster_path_list,
            task_name='project for %s' % country_name)

        people_access_path = os.path.join(
            country_workspace, 'people_access_%s.tif' % country_name)
        people_access_task = task_graph.add_task(
            func=people_access,
            priority=-country_index,
            args=(
                utm_friction_path, utm_population_path, utm_hab_path,
                MAX_TRAVEL_TIME, MAX_TRAVEL_DISTANCE, people_access_path),
            target_path_list=[people_access_path],
            dependent_task_list=[projection_task],
            task_name='calculating people access for %s' % country_name)

    task_graph.close()
    task_graph.join()


def people_access(
        friction_raster_path, population_raster_path, habitat_raster_path,
        max_travel_time, max_travel_distance, target_people_access_path):
    """Construct a people access raster showing where people can reach.

    The people access raster will have a value of population count per pixel
    which can reach that pixel within a cutoff of `max_travel_time` or
    `max_travel_distance`.

    Parameters:
        friction_raster_path (str): path to a raster whose units are
            minutes/meter required to cross any given pixel. Values of 0 are
            treated as impassible.
        population_raster_path (str): path to a per-pixel population count
            raster.
        max_travel_time (float): the maximum amount of time in minutes to
            allow when determining where population can travel to.
        max_travel_distance (float): the maximum distance to allow for when
            determining where population can travel to.
        target_people_access_path (str): raster created by this call that
            will contain the count of population that can reach any given
            pixel within the travel time and travel distance constraints.

    Returns:
        None.

    """
    pygeoprocessing.new_raster_from_base(
        population_raster_path, target_people_access_path, gdal.GDT_Float32,
        [-1], fill_value_list=[-1])
    people_access_raster = gdal.OpenEx(
        target_people_access_path, gdal.OF_RASTER | gdal.GA_Update)
    people_access_band = people_access_raster.GetRasterBand(1)
    people_access_nodata = people_access_band.GetNoDataValue()
    friction_raster_info = pygeoprocessing.get_raster_info(
        friction_raster_path)
    cell_length = friction_raster_info['pixel_size'][0]
    max_travel_distance_in_pixels = max_travel_distance / cell_length
    core_size = int(max_travel_distance_in_pixels*2)
    nx, ny = friction_raster_info['raster_size']
    LOGGER.debug('%d %d', nx, ny)
    window_size = core_size*3
    friction_array = numpy.empty((window_size, window_size))
    population_array = numpy.empty((window_size, window_size))
    habitat_array = numpy.empty((window_size, window_size))
    LOGGER.debug('friction array size: %s', friction_array.shape)

    friction_raster = gdal.OpenEx(
        friction_raster_path, gdal.OF_RASTER)
    friction_band = friction_raster.GetRasterBand(1)
    population_raster = gdal.OpenEx(
        population_raster_path, gdal.OF_RASTER)
    population_band = population_raster.GetRasterBand(1)
    population_nodata = population_band.GetNoDataValue()

    for core_y in range(0, ny, core_size):
        raster_y = core_y - core_size
        raster_y_offset = 0
        raster_win_ysize = window_size
        if raster_y < 0:
            raster_y_offset = abs(raster_y)
            raster_win_ysize += raster_y
            raster_y = 0
        if raster_y + raster_win_ysize > ny:
            raster_win_ysize = ny - raster_y
        core_y_size = core_size
        if core_y + core_y_size > ny:
            core_y_size = ny - core_y
        for core_x in range(0, nx, core_size):
            raster_x = core_x - core_size
            raster_x_offset = 0
            raster_win_xsize = window_size
            if raster_x < 0:
                raster_x_offset = abs(raster_x)
                raster_win_xsize += raster_x
                raster_x = 0
            if raster_x + raster_win_xsize > nx:
                raster_win_xsize = nx - raster_x
            core_x_size = core_size
            if core_x + core_x_size > nx:
                core_x_size = nx - core_x
            friction_array[:] = numpy.nan
            LOGGER.debug(
                '%d:(%d)%d(%d), %d:(%d)%d(%d)',
                raster_y, core_y, raster_win_ysize, raster_y_offset,
                raster_x, core_x, raster_win_xsize, raster_x_offset)

            friction_band.ReadAsArray(
                xoff=raster_x, yoff=raster_y,
                win_xsize=raster_win_xsize, win_ysize=raster_win_ysize,
                buf_obj=friction_array[
                    raster_y_offset:raster_y_offset+raster_win_ysize,
                    raster_x_offset:raster_x_offset+raster_win_xsize])
            population_array[:] = 0.0
            population_band.ReadAsArray(
                xoff=raster_x, yoff=raster_y,
                win_xsize=raster_win_xsize, win_ysize=raster_win_ysize,
                buf_obj=population_array[
                    raster_y_offset:raster_y_offset+raster_win_ysize,
                    raster_x_offset:raster_x_offset+raster_win_xsize])
            total_population = numpy.sum(population_array[
                ~numpy.isclose(population_array, population_nodata)])
            # don't route population where there isn't any
            if total_population < POPULATION_COUNT_CUTOFF:
                continue
            habitat_array[:] = 0.0
            population_band.ReadAsArray(
                xoff=raster_x, yoff=raster_y,
                win_xsize=raster_win_xsize, win_ysize=raster_win_ysize,
                buf_obj=habitat_array[
                    raster_y_offset:raster_y_offset+raster_win_ysize,
                    raster_x_offset:raster_x_offset+raster_win_xsize])
            habitat_amount = numpy.count_nonzero(habitat_array == 1.0)
            if habitat_amount == 0:
                continue

            population_array[
                numpy.isclose(population_array, population_nodata)] = 0.0
            # the nodata value is undefined but will present as 0.
            friction_array[numpy.isclose(friction_array, 0)] = numpy.nan
            # buffer_array[core_y:buffer_ysize, local_x:buffer_xsize]
            population_reach = shortest_distances.find_population_reach(
                friction_array, population_array, cell_length, core_size,
                core_size, core_size, MAX_TRAVEL_TIME, MAX_TRAVEL_DISTANCE)
            LOGGER.debug('population reach size: %s', population_reach.shape)
            LOGGER.debug(
                'core_y_size %d, core_x_size %d '
                'core_x %d, core_y %d', core_y_size, core_x_size, core_x,
                core_y)
            # mask by habitat -- set to nodata where there's no habitat
            population_reach[habitat_array[
                core_size:2*core_size, core_size:2*core_size] != 0] = (
                    people_access_nodata)
            people_access_band.WriteArray(
                population_reach[0:core_y_size, 0:core_x_size],
                xoff=core_x, yoff=core_y)


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


def extract_and_project_feature(
        vector_path, feature_id, projection_wkt, target_vector_path,
        target_complete_token_path):
    """Make a local projection of a single feature in a vector.

    Parameters:
        vector_path (str): base vector in WGS84 coordinates.
        feature_id (int): FID for the feature to extract.
        projection_wkt (str): projection wkt code to project feature to.
        target_gpkg_vector_path (str): path to new GPKG vector that will
            contain only that feature.
        target_complete_token_path (str): path to a file that is created if
             the function successfully completes.

    Returns:
        None.

    """
    base_vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    base_layer = base_vector.GetLayer()
    feature = base_layer.GetFeature(feature_id)
    geom = feature.GetGeometryRef()

    epsg_srs = osr.SpatialReference()
    epsg_srs.ImportFromWkt(projection_wkt)

    base_srs = base_layer.GetSpatialRef()
    base_to_utm = osr.CoordinateTransformation(base_srs, epsg_srs)

    # clip out watershed to its own file
    # create a new shapefile
    if os.path.exists(target_vector_path):
        os.remove(target_vector_path)
    driver = ogr.GetDriverByName('GPKG')
    target_vector = driver.CreateDataSource(
        target_vector_path)
    target_layer = target_vector.CreateLayer(
        os.path.splitext(os.path.basename(target_vector_path))[0],
        epsg_srs, ogr.wkbMultiPolygon)
    layer_defn = target_layer.GetLayerDefn()
    feature_geometry = geom.Clone()
    base_feature = ogr.Feature(layer_defn)
    feature_geometry.Transform(base_to_utm)
    base_feature.SetGeometry(feature_geometry)
    target_layer.CreateFeature(base_feature)
    target_layer.SyncToDisk()
    geom = None
    feature_geometry = None
    base_feature = None
    target_layer = None
    target_vector = None
    base_layer = None
    base_vector = None
    with open(target_complete_token_path, 'w') as token_file:
        token_file.write(str(datetime.datetime.now()))


def length_of_degree(lat):
    """Calculate the length of a degree in meters."""
    m1 = 111132.92
    m2 = -559.82
    m3 = 1.175
    m4 = -0.0023
    p1 = 111412.84
    p2 = -93.5
    p3 = 0.118
    lat_rad = lat * numpy.pi / 180
    latlen = (
        m1 + m2 * numpy.cos(2 * lat_rad) + m3 * numpy.cos(4 * lat_rad) +
        m4 * numpy.cos(6 * lat_rad))
    longlen = abs(
        p1 * numpy.cos(lat_rad) + p2 * numpy.cos(3 * lat_rad) + p3 * numpy.cos(5 * lat_rad))
    return max(latlen, longlen)


if __name__ == '__main__':
    main()
