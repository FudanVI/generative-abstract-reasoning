# Towards Generative Abstract Reasoning: Completing Raven’s Progressive Matrix via Rule Abstraction and Selection

This repository is the implementation of the paper "Towards Generative Abstract Reasoning: Completing Raven’s Progressive Matrix via Rule Abstraction and Selection".

## Requirements

To install all requirements, run

```setup
conda env create -f environment.yml
```

## Datasets

1. Prepare the [RAVEN](https://github.com/WellyZhang/RAVEN) and [I-RAVEN](https://github.com/husheng12345/SRAN) datasets

2. Modify the dataset root in `dataset/build_RAVEN.sh` and `dataset/build_IRAVEN.sh`

   ```setup
   python dataset/RAVEN/create_RAVEN.py \
     --dataset_root '<Your path to the root of RAVEN dataset>'\
     --size 64 \
     --output_path './cache'
   ```
   ```setup
   python dataset/IRAVEN/create_IRAVEN.py \
     --dataset_root '<Your path to the root of I-RAVEN dataset>'\
     --size 64 \
     --output_path './cache'
   ```

3. In the root, run

   ```setup
   sh create_dataset.sh
   ```

## Training

Change the conda environment

```
conda activate iclr2024-code
```

To train RAISE on the RAVEN dataset, run

```
python train.py --dataset RAVEN --image_type [CFG_NAME] \
	--exp_name RAISE --model model_raise --gpu [GPU_ID]
```

`[GPU_ID]` controls the device  (>=0 for gpu and -1 for cpu) to run the test program

`[CFG_NAME]` controls the image configuration used in training

| Configuration  |                CFG_NAME                |
| :------------: | :------------------------------------: |
|     Center     |             center_single              |
|      L-R       | left_center_single_right_center_single |
|      U-D       |  up_center_single_down_center_single   |
|      O-IC      |   in_center_single_out_center_single   |
|      O-IG      |  in_distribute_four_out_center_single  |
| 2$\times$2Grid |            distribute_four             |
| 3$\times$3Grid |            distribute_nine             |

## Evaluation

To evaluate RAISE on all configurations of RAVEN/I-RAVEN, run

```
sh test.sh [GPU_ID]
```
`[GPU_ID]` controls the device  (>=0 for gpu and -1 for cpu) to run the test program.

