DATA_DIR=$1

results=./results/roadnet/test_latest/images

for i in `seq 1 14`
#for i in 1
do
    python3 ./houston/post-proc/merge_predictions.py --results_dir ${results} --image_number $i
done