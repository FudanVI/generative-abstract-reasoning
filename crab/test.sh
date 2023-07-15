gpu=0

python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type left_center_single_right_center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type up_center_single_down_center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type in_center_single_out_center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type in_distribute_four_out_center_single_uniform --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type distribute_four_uniform --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset RAVEN \
  --image_type distribute_nine_uniform --gpu ${gpu} --batch_size 512


python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type left_center_single_right_center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type up_center_single_down_center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type in_center_single_out_center_single --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type in_distribute_four_out_center_single_uniform --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type distribute_four_uniform --gpu ${gpu} --batch_size 512
python test.py --exp_name crab --model model_score --dataset I-RAVEN \
  --image_type distribute_nine_uniform --gpu ${gpu} --batch_size 512
