gpu="0"
MODEL_ZOO_DIR="/scratch/guanhong/BAIT-ModelZoo-Alpaca/models"
CACHE_DIR="/scratch/guanhong/gy_bait_model_zoo/base_models"
# MODEL_ID="id-0007"
RUN_NAME="bait-alpaca-test"
OUTPUT_DIR="result"
DATA_DIR="data"

CUDA_VISIBLE_DEVICES=$gpu python -m bait.main \
    --model_zoo_dir $MODEL_ZOO_DIR \
    --cache_dir $CACHE_DIR \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME 
# done

