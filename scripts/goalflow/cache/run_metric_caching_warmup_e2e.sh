# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

SPLIT=mini
SCENE_FILTER=warmup_test_e2e
CACHE_TO_SAVE='/media/farrosalferro/College/research/AD/navsim/exp/goalflow_metric_cache' #set your metric cache path to save

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=warmup_test_e2e \
metric_cache_path=$CACHE_TO_SAVE \
train_test_split.scene_filter.frame_interval=1  \
sensor_blobs_path=${OPENSCENE_DATA_ROOT}/sensor_blobs/${split}
# split=$SPLIT \
