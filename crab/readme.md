# Abstracting Concept-Changing Rules for Solving Raven’s Progressive Matrix Problems

This repository is the implementation of the paper "Abstracting Concept-Changing Rules for Solving Raven’s Progressive Matrix Problems".

## Requirements

Install requirements

```setup
conda env create -f environment.yml
```

and switch to the created environment

```
conda activate crab
```

## Prepare Datasets

### RAVEN

1. Generate or download RAVEN dataset through [this repository](https://github.com/WellyZhang/RAVEN).

2. Modify `--dataset_root` in `dataset/build_RAVEN.sh` accordingly:

```setup
python dataset/RAVEN/create_RAVEN.py \
    --dataset_root '<root of the RAVEN dataset>'\
    --size 64 \
    --output_path 'cache'
```

### I-RAVEN

1. Generate I-RAVEN dataset through [this repository](https://github.com/husheng12345/SRAN).

2. Modify `--dataset_root` in `dataset/build_IRAVEN.sh` accordingly:

```setup
python dataset/IRAVEN/create_IRAVEN.py \
    --dataset_root '<root of the I-RAVEN dataset>'\
    --size 64 \
    --output_path 'cache'
```

### Generate cache files

In the root, execute the script

```setup
    ./create_dataset.sh
```

## Training

To train the model in the paper, run the command

```
python train.py --dataset RAVEN --image_type [Imgae Config] \
	--exp_name crab --model model_score --gpu [GPU ID]
```

or execute the script

```setup
./train.sh [Config Alias] [GPU ID]
```

For different image configurations in this paper:

1. Center: [Image Config] == 'center_single', [Config Alias] == 'cs'
2. L-R: [Image Config] == 'left_center_single_right_center_single', [Config Alias] == 'lr'
3. U-D: [Image Config] == 'up_center_single_down_center_single', [Config Alias] == 'ud'
4. O-IC: [Image Config] == 'in_center_single_out_center_single', [Config Alias] == 'oic'
5. O-IG: [Image Config] == 'in_distribute_four_out_center_single_uniform', [Config Alias] == 'oig'
6. 2*2Grid: [Image Config] == 'distribute_four_uniform', [Config Alias] == '22'
7. 3*3Grid: [Image Config] == 'distribute_nine_uniform', [Config Alias] == '33'

## Evaluation

To evaluate the model, execute the script

```setup
./test.sh
```
