import argparse
from osgeo import gdal
from osgeo import osr
import pygeoprocessing

if __name__ == '__main__':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('lat', type=float)
    parser.add_argument('lng', type=float)
    parser.add_argument('country_name')
    args = parser.parse_args()

    base_ref = osr.SpatialReference()
    base_ref.ImportFromWkt(osr.SRS_WKT_WGS84_LAT_LONG)

    target_ref = osr.SpatialReference()
    target_ref.ImportFromWkt(world_eckert_iv_wkt)

    base_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # Create a coordinate transformation
    transformer = osr.CreateCoordinateTransformation(base_ref, target_ref)
    back_transformer = osr.CreateCoordinateTransformation(target_ref, base_ref)

    trans_x, trans_y, _ = transformer.TransformPoint(args.lng, args.lat)
    print(f'({trans_x}, {trans_y})')
    back_lng, back_lat, _ = back_transformer.TransformPoint(trans_x, trans_y)
    print(f'({back_lat}, {back_lng})')

    world_borders_vector = gdal.OpenEx(
        'TM_WORLD_BORDERS-0.3_simplified_md5_47f2059be8d4016072aa6abe77762021.gpkg',
        gdal.OF_VECTOR)
    world_borders_layer = world_borders_vector.GetLayer()
    for country_feature in world_borders_layer:
        if country_feature.GetField("NAME").lower() == args.country_name.lower():
            country_geometry = country_feature.GetGeometryRef()
            country_bb = [
                country_geometry.GetEnvelope()[i] for i in [0, 2, 1, 3]]
            print(country_bb)
            transformbb = pygeoprocessing.transform_bounding_box(
                country_bb, osr.SRS_WKT_WGS84_LAT_LONG,
                world_eckert_iv_wkt, edge_samples=11)
            print(f'transformbb: {transformbb}')

