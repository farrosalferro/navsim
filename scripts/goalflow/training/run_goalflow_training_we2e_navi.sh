# LD_LIBRARY_PATH="/usr/local/cuda/lib64"
# export CUDA_VISIBLE_DEVICES=1
# export HYDRA_FULL_ERROR=1


FEATURE_CACHE='/media/farrosalferro/College/research/AD/navsim/exp/goalflow_dataset_cache' # set your feature_cache path
V99_PRETRAINED_PATH=/media/farrosalferro/College/research/AD/navsim/goalflow_data/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth
CHECKPOINT_PATH=''
VOC_PATH=/media/farrosalferro/College/research/AD/navsim/goalflow_data/cluster_points_8192_.npy
DAC_LABEL_PATH=/media/farrosalferro/College/research/AD/navsim/checkpoint/goalflow/dac_scores
FREEZE_PERCEPTION=True # you can choose False and increase batch_size if the GPU are sufficient

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_goalflow.py \
agent=goalflow_agent_navi \
experiment_name=a_train_navi \
train_test_split=warmup_test_e2e \
split=mini \
cache_path=$FEATURE_CACHE \
trainer.params.max_epochs=10 \
agent.config.training=True \
agent.config.freeze_perception=$FREEZE_PERCEPTION \
agent.config.tf_d_model=1024 \
agent.config.bev_semantic_weight=10.0 \
agent.config.dac_score_weight=0.005 \
agent.config.im_score_weight=1.0 \
dataloader.params.batch_size=1 \
use_cache_without_dataset=True \
dac_label_path=$DAC_LABEL_PATH \
agent.config.v99_pretrained_path=$V99_PRETRAINED_PATH \
agent.checkpoint_path=$CHECKPOINT_PATH \
agent.config.voc_path=$VOC_PATH