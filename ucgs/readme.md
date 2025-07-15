# Beyond Task-Specific Reasoning: A Unified Conditional Generative Framework for Abstract Visual Reasoning

This repository is the implementation of the paper "Beyond Task-Specific Reasoning: A Unified Conditional Generative Framework for Abstract Visual Reasoning".

This repository is still under development. We have released the training and evaluation code for UCGS-T. More detailed instructions and codes for baseline models will be made available as soon as possible.

## Datasets

1. Prepare the RAVEN, PGM, G1SET, VAP and SVRT datasets

2. Modify the dataset roots in `create_dataset.sh`, for example:

   ```setup
   python dataset/RAVEN/create_cache.py \
     --input_path '<Your path to the root of RAVEN dataset>' \
     --output_path '<Your cache root to the RAVEN dataset>'
   ```

3. Run

   ```setup
   bash create_dataset.sh
   ```

4. Modify the cache roots in `config/main.yaml`, for example:
   
   ```setup
   RAVEN:
     num_cell: 9
     src: <Your cache root to the RAVEN dataset>
     instances: [base]
   ```

## Training

To train the VQVAE backbone on the RAVEN and PGM datasets, run

```
python train.py --model backbone_vqvae --exp_name vqvae --dataset RAVEN,PGM --gpu [GPU_ID]
```

To train UCGS-T on the RAVEN and PGM datasets, run

```
python train.py --model model_ucgst --exp_name ucgst --dataset RAVEN,PGM --gpu [GPU_ID]
```

`[GPU_ID]` controls the device  (>=0 for gpu and -1 for cpu) to run the code

## Evaluation

To evaluate UCGS-T on ID tasks, ID-ZS tasks and OOD-ZS tasks, run

```
python test.py --model model_ucgst --exp_name ucgst --gpu [GPU_ID]
```
