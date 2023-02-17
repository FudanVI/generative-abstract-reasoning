python dataset/CRPM/create_polygon.py \
  --num_train 10000 --num_val 1000 --num_test 2000 --name triangle -n 3 \
  --output_path ./cache

python dataset/CRPM/create_complex_polygon.py \
  --num_train 10000 --num_val 1000 --num_test 2000 --name complex_triangle -n 3 \
  --output_path ./cache

python dataset/CRPM/create_circle.py \
  --num_train 10000 --num_val 1000 --num_test 2000 --name circle \
  --output_path ./cache

python dataset/CRPM/create_complex_circle.py \
  --num_train 10000 --num_val 1000 --num_test 2000 --name complex_circle \
  --output_path ./cache