# Specify the config file path and the GPU devices to use
# export CUDA_VISIBLE_DEVICES=0,1

# Specify the config file path
export XFL_CONFIG=./train/config/config_densedit.yaml

# # Specify the WANDB API key
# export WANDB_API_KEY=''

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m src.train.train