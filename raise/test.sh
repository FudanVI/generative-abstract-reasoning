for image_type in center_single left_center_single_right_center_single up_center_single_down_center_single \
  in_center_single_out_center_single in_distribute_four_out_center_single \
  distribute_four distribute_nine
do
  python test.py --exp_name RAISE --model model_raise --dataset RAVEN \
    --image_type $image_type --gpu $1 --batch_size 512
  python test.py --exp_name RAISE --model model_raise --dataset I-RAVEN \
    --image_type $image_type --gpu $1 --batch_size 512
done
