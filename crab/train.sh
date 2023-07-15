image_config=${1}
gpu=${2}

case $image_config in
  cs)
    name=center_single;;
  lr)
    name=left_center_single_right_center_single;;
  ud)
    name=up_center_single_down_center_single;;
  oic)
    name=in_center_single_out_center_single;;
  oig)
    name=in_distribute_four_out_center_single_uniform;;
  22)
    name=distribute_four_uniform;;
  33)
    name=distribute_nine_uniform;;
  *) echo "This configuration is not available.";;
esac

echo $name
echo $gpu

python train.py --dataset RAVEN --image_type $name --model model_score --exp_name crab --gpu $gpu