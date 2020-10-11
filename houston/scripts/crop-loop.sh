DATA_DIR=$1

save_folder=../../datasets/Houston-Dataset

mkdir ${save_folder}/test_image
mkdir ${save_folder}/test_segment
mkdir ${save_folder}/test_edge
mkdir ${save_folder}/test_centerline

# test
for i in `seq 1 14`
#for i in 7
do
    python3 ../../tools/image_crop.py --image_file "${DATA_DIR}/$i/Houston-$i.tif" --save_path ${save_folder}/test_image --step 256
    python3 ../../tools/image_crop.py --image_file "${DATA_DIR}/$i/segmentation.png" --save_path ${save_folder}/test_segment --step 256
    python3 ../../tools/image_crop.py --image_file "${DATA_DIR}/$i/edge.png" --save_path ${save_folder}/test_edge --step 256
    python3 ../../tools/image_crop.py --image_file "${DATA_DIR}/$i/centerline.png" --save_path ${save_folder}/test_centerline --step 256
done