"""Distance to habitat with a friction layer."""
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
    'friction_surface': 'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/friction_surface_2015_v1.0-002_md5_166d17746f5dd49cfb2653d721c2267c.tif',
    'population_layer': r'https://storage.googleapis.com/ecoshard-root/lspop2017_md5_86d653478c1d99d4c6e271bad280637d.tif',
    'world_borders': r'https://storage.googleapis.com/ecoshard-root/critical_natural_capital/TM_WORLD_BORDERS-0.3_simplified_md5_47f2059be8d4016072aa6abe77762021.gpkg'
}

WORKSPACE_DIR = 'workspace_dist_to_hab_with_friction'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
TARGET_NODATA = -1

# max travel time in minutes, basing off of half of a travel day (roundtrip)
MAX_TRAVEL_TIME = 4*60  # minutes

# max travel distance to cutoff simulation
MAX_TRAVEL_DISTANCE = 40000


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

    friction_raster_info = pygeoprocessing.get_raster_info(
        ecoshard_path_map['friction_surface'])
    for country_feature in world_borders_layer:
        country_name = country_feature.GetField('NAME')
        fid = country_feature.GetFID()
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
        country_workspace = os.path.join(WORKSPACE_DIR, country_name)
        try:
            os.makedirs(country_workspace)
        except OSError:
            pass
        country_vector_path = os.path.join(
            country_workspace, '%s.gpkg' % country_name)
        extract_country_task = task_graph.add_task(
            func=extract_and_project_feature,
            args=(
                ecoshard_path_map['world_borders'], fid, epsg_code,
                country_vector_path),
            target_path_list=[country_vector_path],
            task_name='make local watershed for %s' % country_name)
        extract_country_task.join()

        base_raster_path_list = [
            ecoshard_path_map['friction_surface'],
            ecoshard_path_map['population_layer'],
        ]
        wgs84_raster_path_list = [
            os.path.join(
                country_workspace, 'wgs84_%s_friction.tif' % country_name),
            os.path.join(
                country_workspace, 'wgs84_%s_population.tif' % country_name)
        ]
        clip_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
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
        utm_raster_path_list = [utm_friction_path, utm_population_path]

        m_per_deg = length_of_degree(centroid_geom.GetY())
        target_pixel_size = (
            m_per_deg*friction_raster_info['pixel_size'][0],
            m_per_deg*friction_raster_info['pixel_size'][1])
        LOGGER.debug(target_pixel_size)
        # this will project values outside to 0 since there's not a nodata
        # value defined
        projection_task = task_graph.add_task(
            func=pygeoprocessing.align_and_resize_raster_stack,
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
            country_workspace, 'people_access.tif')
        people_access(
            utm_friction_path, utm_population_path, MAX_TRAVEL_TIME,
            MAX_TRAVEL_DISTANCE, people_access_path)

    task_graph.close()


def people_access(
        friction_raster_path, population_raster_path,
        max_travel_time, max_travel_distance, target_people_access_path):
    """Construct a people access raster showing where people can reach.

    The people access raster will have a value of population count per pixel
    which can reach that pixel within a cutoff of `max_travel_time` or
    `max_travel_distance`.

    Parameters:
        friction_raster_path (str): path to a raster whose units are
            meters/minute required to cross any given pixel. Values of 0 are
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
        population_raster_path, target_people_access_path, gdal.GDT_Int32,
        [-1], fill_value_list=[-1])
    friction_raster_info = pygeoprocessing.get_raster_info(
        friction_raster_path)
    max_travel_distance_in_pixels = (
        max_travel_distance / friction_raster_info['pixel_size'][0])
    LOGGER.debug(max_travel_distance_in_pixels*4)
    window_size = int(max_travel_distance_in_pixels*2)
    buffer_size = int(max_travel_distance_in_pixels)
    nx, ny = friction_raster_info['raster_size']

    # `window_i/j` is the upper left hand coordinate of the valid travel
    # distance block that will be calculated
    # `window_size` is the number of pixels w/h that will have valid travel
    # calculations from the `window_i/j` coordinate
    # `buffer_i/j` is the upper left hand corner of the window that will be
    # read in to process all pairs least cost paths
    # buffer size is the number of pixels to pad all around the window
    buffer_array = numpy.empty((
        window_size + 2*buffer_size, window_size + 2*buffer_size))
    for window_j in range(0, ny, window_size):
        buffer_j = window_j - buffer_size
        buffer_ysize = window_size + 2*buffer_size
        local_j = 0
        if buffer_j < 0:
            buffer_ysize += buffer_j
            local_j -= buffer_j
            buffer_j = 0
        if buffer_j + buffer_ysize >= ny:
            buffer_ysize = ny - buffer_j
        for window_i in range(0, nx, window_size):
            buffer_i = window_i - buffer_size
            local_i = 0
            buffer_xsize = window_size + 2*buffer_size
            if buffer_i < 0:
                buffer_xsize += buffer_i
                local_i -= buffer_i
                buffer_i = 0
            if buffer_i + buffer_xsize >= nx:
                buffer_xsize = nx - buffer_i

            buffer_array[:] = numpy.inf
            # what are the buffer array bounds?
            # 0,0 to buffer_xsize, buffer_ysize is the default
            LOGGER.debug(window_size+buffer_size*2)
            LOGGER.debug(
                '%d: %d-%d, %d: %d-%d .. %s', window_i, local_i, buffer_xsize,
                window_j, local_j, buffer_ysize, buffer_array.shape)
            #buffer_array[local_j:buffer_ysize, local_i:buffer_xsize]


    return
    # extract out that country layer and reproject to a UTM zone.
    n_size = 200
    shortest_distances.find_shortest_distances(

        friction_raster_path, population_raster_path, target_people_access_path
        (ecoshard_path_map['friction_surface'], 1),
        10000, 10000, n_size, n_size)


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
        vector_path, feature_id, epsg_code, target_vector_path):
    """Make a local projection of a single feature in a vector.

    Parameters:
        vector_path (str): base vector in WGS84 coordinates.
        feature_id (int): FID for the feature to extract.
        epsg_code (str): EPSG code to project feature to.
        target_gpkg_vector_path (str): path to new GPKG vector that will
            contain only that feature.

    Returns:
        None.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    feature = layer.GetFeature(feature_id)
    geom = feature.GetGeometryRef()

    epsg_srs = osr.SpatialReference()
    epsg_srs.ImportFromEPSG(epsg_code)

    base_srs = layer.GetSpatialRef()
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
        epsg_srs, ogr.wkbPolygon)
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
