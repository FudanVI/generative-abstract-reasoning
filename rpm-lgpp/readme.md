# Raven’s Progressive Matrices Completion with Latent Gaussian Process Priors (LGPP)

This is the code of the paper "Raven’s Progressive Matrices Completion with Latent Gaussian Process Priors".

## Dependencies

- PyTorch == 1.4
- pyyaml == 5.3
- pillow == 7.0.0
- numpy == 1.18.1
- six == 1.14.0

Python verion 3.7.6

## Datasets

Build the Polygon and Circle datasets:
```
./build_datasets.sh
```

## Training

Train model for the specific dataset (e.g., triangle-instanced Polygon dataset with 5000 training samples)
```
python train.py --exp_name triangle_5000 --gpu 0 --dataset triangle_5000
```
Parameter `--exp_name` is your custom experiment name and `--dataset` is the dataset used in the training phase.

## Experiments
Estimate models with MSE scores
```
python test.py --exp_name triangle_5000 --gpu 0 --dataset triangle_50000
```
Estimate models with disentanglement metrics
```
python evaluate.py --exp_name triangle_5000 --gpu 0 --dataset triangle_50000
```
In the experiments, we test the models in the datasets with 50000 sample.
