# Compositional Law Parsing with Latent Random Functions

This repository is the implementation of the paper "Compositional Law Parsing with Latent Random Functions".

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

## Dataset

1. Download MPI3D dataset from [here](https://github.com/rr-learning/disentanglement_dataset).

2. Unzip MPI3D dataset and modify the dataset root in `dataset/build_MPI3D.sh`:

   ```setup
   python dataset/MPI3D/create_MPI3D.py \
     --dataset_root '<Your path to the .npz files of MPI3D dataset>'\
     --size 64 \
     --output_path './cache'
   ```

3. In the root, run

   ```setup
   ./create_dataset.sh
   ```

## Training

Change the conda environments:

```
conda activate iclr2023-code
```

To train the model in the paper, run this command:

```
python train.py --dataset [Dataset Name] --image_type [Imgae Type] \
	--exp_name CLAP --model model_clap_np --gpu 0
```

For datasets in the paper:

1. BoBa-1: [Dataset Name] == 'balls' and [Image Type] == '1_obj'
2. BoBa-2: [Dataset Name] == 'balls' and [Image Type] == '2_obj'
3. CRPM-T: [Dataset Name] == 'CRPM' and [Image Type] == 'triangle'
4. CRPM-DT: [Dataset Name] == 'CRPM' and [Image Type] == 'complex_triangle'
5. CRPM-C: [Dataset Name] == 'CRPM' and [Image Type] == 'circle'
6. CRPM-DC: [Dataset Name] == 'CRPM' and [Image Type] == 'complex_circle'
7. MPI3D: [Dataset Name] == 'MPI3D' and [Image Type] == 'real_complex'

## Evaluation

To evaluate models on all datasets, run this command:

```
./test.sh
```
