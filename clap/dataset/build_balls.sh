for dataset in 1_obj 2_obj
do
  python dataset/balls/create_balls.py \
    --config_path 'dataset/balls/config.yaml' \
    --shape_path 'dataset/balls/shapes' \
    --output_path './cache' \
    --dataset ${dataset}
done