sudo docker run -it --gpus all -d \
    -v ~/Documents/CEIA/offline_to_online/CORL/checkpoints/:/CORL/checkpoints \
    -v ~/Documents/CEIA/offline_to_online/CORL/datasets/:/datasets \
    -v ~/Documents/CEIA/offline_to_online/CORL/configs/:/CORL/configs \
    luanamartins/corl \
    python -m algorithms.offline.any_percent_bc --config_path="configs/offline/bc/go2/footstand_medium.yaml" --device "cuda