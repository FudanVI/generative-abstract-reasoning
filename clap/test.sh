gpu=0

python test.py --exp_name CLAP --model model_clap_np --dataset balls --image_type 1_obj --gpu ${gpu}
python test.py --exp_name CLAP --model model_clap_np --dataset balls --image_type 2_obj --gpu ${gpu}
python test.py --exp_name CLAP --model model_clap_np --dataset MPI3D --image_type real_complex --gpu ${gpu}
python test.py --exp_name CLAP --model model_clap_np --dataset CRPM --image_type triangle --gpu ${gpu}
python test.py --exp_name CLAP --model model_clap_np --dataset CRPM --image_type circle --gpu ${gpu}
python test.py --exp_name CLAP --model model_clap_np --dataset CRPM --image_type complex_triangle --gpu ${gpu}
python test.py --exp_name CLAP --model model_clap_np --dataset CRPM --image_type complex_circle --gpu ${gpu}
