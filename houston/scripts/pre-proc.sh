DATA_DIR=$1

for i in `seq 1 14`
do
    python3 ../pre-proc/create_osm_query.py --data_dir "${DATA_DIR}/$i"
    wget --post-file="${DATA_DIR}/$i/query.txt"  http://overpass-api.de/api/interpreter --output-document="${DATA_DIR}/$i/centerline.osm"
    osmtogeojson "${DATA_DIR}/$i/centerline.osm" > "${DATA_DIR}/$i/centerline.geojson" 
    python3 ../pre-proc/pre-proc.py --data_dir "${DATA_DIR}/$i"
done