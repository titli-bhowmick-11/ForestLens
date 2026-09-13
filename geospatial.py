import geopandas as gpd

def read_kml(kml_path):

    gdf = gpd.read_file(kml_path)

    area = gdf.to_crs(epsg=3857).area.sum()

    hectares = area / 10000

    return hectares