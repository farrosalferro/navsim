# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"

SPLIT=mini
METRIC_CACHE='/media/farrosalferro/College/research/AD/navsim/exp/goalflow_metric_cache' # metric cache in run_dataset_cache_trainval you set
VOC_PATH=/media/farrosalferro/College/research/AD/navsim/goalflow_data/cluster_points_8192_.npy
DAC_LABEL_PATH='/media/farrosalferro/College/research/AD/navsim/checkpoint/goalflow/dac_scores' # add the dac label path you want to storage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dac_score.py \
agent=goalflow_agent_traj \
experiment_name=dac_score \
train_test_split=warmup_test_e2e \
split=$SPLIT \
metric_cache_path=$METRIC_CACHE \
agent.config.voc_path=$VOC_PATH \
dac_label_path=$DAC_LABEL_PATH