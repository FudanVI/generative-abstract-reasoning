for size in 50 100 500 1000 5000 10000 50000
do
  python dataset/create_polygon.py --num_train ${size} --num_val `expr ${size} / 10` \
  --num_test `expr ${size} / 5` --name triangle_${size} -n 3
done

for size in 50 100 500 1000 5000 10000 50000
do
  python dataset/create_circle.py --num_train ${size} --num_val `expr ${size} / 10` \
  --num_test `expr ${size} / 5` --name circle_${size}
done
