# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export HYDRA_FULL_ERROR=1

SPLIT=mini
METRIC_CACHE="/media/farrosalferro/College/research/AD/navsim/exp/goalflow_metric_cache" # set your metric path 
TRAJS_CACHE="/media/farrosalferro/College/research/AD/navsim/exp/a_test_release/2025.04.01.11.25.36/lightning_logs/version_0/trajs" # set your trajectories path

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_trajs_adapted.py \
agent=goalflow_agent_traj \
agent.checkpoint_path=/media/farrosalferro/College/research/AD/navsim/lightning_logs/version_0/trajectory_planning/epoch_9-step_1850.ckpt \
agent.config.tf_d_model=1024 \
experiment_name=a_test_release_result \
train_test_split=warmup_test_e2e \
metric_cache_path=$METRIC_CACHE \
trajs_cache_path=$TRAJS_CACHE \
sensor_blobs_path=media/farrosalferro/College/research/AD/navsim/dataset/sensor_blobs/mini \
worker=single_machine_thread_pool 
# split=$SPLIT \