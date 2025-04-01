# LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export CUDA_VISIBLE_DEVICES=2

FEATURE_CACHE=/media/farrosalferro/College/research/AD/navsim/exp/goalflow_dataset_cache
VOC_PATH=/media/farrosalferro/College/research/AD/navsim/goalflow_data/cluster_points_8192_.npy
CHECKPOINT_PATH=/media/farrosalferro/College/research/AD/navsim/lightning_logs/version_0/navi/epoch_4-step_1850.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_trajs.py \
agent=goalflow_agent_navi \
experiment_name=a_test_release_navi \
train_test_split=warmup_test_e2e \
split=mini \
cache_path=$FEATURE_CACHE \
agent.config.generate='goal_score' \
agent.config.voc_path=$VOC_PATH \
agent.config.training=False \
agent.config.tf_d_model=1024 \
dataloader.params.batch_size=4 \
use_cache_without_dataset=True \
agent.checkpoint_path=$CHECKPOINT_PATH