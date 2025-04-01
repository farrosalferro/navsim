# export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# The trajectory_sampling.time_horizon in trainval is 5
CACHE_TO_SAVE='/media/farrosalferro/College/research/AD/navsim/exp/goalflow_dataset_cache' #set your feature cache path to save

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_goalflow.py \
agent=goalflow_agent_traj \
agent.trajectory_sampling.time_horizon=5 \
experiment_name=a_goalflow_test_cache \
cache_path=$CACHE_TO_SAVE \
train_test_split=warmup_test_e2e \
split=mini \
worker=single_machine_thread_pool 