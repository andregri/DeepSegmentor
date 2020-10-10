from osgeo import gdal,ogr,osr
import numpy as np
import argparse
from pathlib import Path

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            #print(x,y)
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


parser = argparse.ArgumentParser(description="Generate the query txt file to download osm data")
parser.add_argument('--data_dir', type=str, default="/home/andrea/Downloads/Final RGB HR Imagery/5", help="directory where tif image is located")

args = parser.parse_args()
data_dir = Path(args.data_dir)
tif_file = [img for img in data_dir.rglob('UH_NAD*.tif')][0]
raster = tif_file.as_posix()
ds=gdal.Open(raster)

gt=ds.GetGeoTransform()
cols = ds.RasterXSize
rows = ds.RasterYSize
ext=GetExtent(gt,cols,rows)

src_srs=osr.SpatialReference()
src_srs.ImportFromWkt(ds.GetProjection())
#tgt_srs=osr.SpatialReference()
#tgt_srs.ImportFromEPSG(4326)
tgt_srs = src_srs.CloneGeogCS()

geo_ext = ReprojectCoords(ext,src_srs,tgt_srs)
print(geo_ext)

longs = [latlon[1] for latlon in geo_ext]
lon_min = np.array(longs).min()
lon_max = np.array(longs).max()

lats = [latlon[0] for latlon in geo_ext]
lat_min = np.array(lats).min()
lat_max = np.array(lats).max()

bbox = "{},{},{},{}".format(lat_min, lon_min, lat_max, lon_max)
print(bbox)

query_template = """
[out:xml][timeout:25];\n
(\n
  //{{bbox}}\n
  way["highway"="residential"]({{bbox}});\n
  way["highway"="secondary"]({{bbox}});\n
  way["highway"="tertiary"]({{bbox}});\n
);\n
\n
// print results\n
out body;\n
>;\n
out skel qt;
"""

out_path = (data_dir / 'query.txt').as_posix()
with open (out_path, 'w') as out_file:
    for line in query_template.split('\n'):
        out_file.writelines(line.replace('{{bbox}}', bbox) + '\n')