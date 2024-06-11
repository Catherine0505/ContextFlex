# ContextFlex: Joint Segmentation of Multiple Biomedical Images

This folder provides code for ContextFlex as part of the supplemental material. 

We provide training and evaluation code for the MNIST experiment

## Training 

To train ContextFlex on MNIST

```
export PYTHONPATH="<path_to_contextflex>"

cd <path_to_contextflex>

python scripts/multiseg.py \
--aggregate_type mean --fusion_type append --features <list_of_feature_sizes> --seed <seed_num> \
--dataset_name MNIST --root_folder <path_to_save_data> --num_classes 1 \
--corrupt --num_corruptions 8 --do_random_noise --noise_ratio 0.9 \
--pad 2 \
--num_workers 4 \
--loss_function mean_softdice_loss mean_ce_loss \
--group_size 2 8 --grouping_criterion <image/digit/similarity> \
--lr <lr> --train_batch_size <batch_size> --val_batch_size <val_batch_size> --max_val_groups 200 \
--save_root_folder <path_to_save_model>
```

To train baseline on MNIST without set average

```
export PYTHONPATH="<path_to_contextflex>"

cd <path_to_contextflex>

python scripts/multiseg.py \
--aggregate_type none --fusion_type none --features <list_of_feature_sizes> --seed <seed_num> \
--dataset_name MNIST --root_folder <path_to_save_data> --num_classes 1 \
--corrupt --num_corruptions 8 --do_random_noise --noise_ratio 0.9 \
--pad 2 \
--num_workers 4 \
--loss_function mean_softdice_loss mean_ce_loss \
--group_size 2 8 --grouping_criterion <image/digit/similarity> \
--lr <lr> --train_batch_size <batch_size> --val_batch_size <val_batch_size> --max_val_groups 200 \
--save_root_folder <path_to_save_model>
```

To train baseline on MNIST with set average

```
export PYTHONPATH="<path_to_contextflex>"

cd <path_to_contextflex>

python scripts/multiseg.py \
--aggregate_type none --fusion_type none --features <list_of_feature_sizes> --seed <seed_num> \
--dataset_name MNIST --root_folder <path_to_save_data> --num_classes 1 \
--corrupt --num_corruptions 8 --do_random_noise --noise_ratio 0.9 \
--pad 2 \
--num_workers 4 \
--loss_function mean_softdice_loss mean_ce_loss \
--group_size 2 8 --grouping_criterion <image/digit/similarity> --average_group \
--lr <lr> --train_batch_size <batch_size> --val_batch_size <val_batch_size> --max_val_groups 200 \
--save_root_folder <path_to_save_model>
```

## Evaluation

To evaluate all trained models, both ContextFlex and baseline, with no set average

```
cd <path_to_contextflex>

python scripts/evaluate.py \
--save_root_folder <path_to_save_model (same as training script)> \
--data_root_folder <path_to_saved_data (same as training script)> \
--grouping_criterion <image/digit/similarity>
```

To evaluate trained baselines, with set average

```
cd <path_to_contextflex>

python scripts/evaluate.py \
--save_root_folder <path_to_save_model (same as training script)> \
--data_root_folder <path_to_saved_data (same as training script)> \
--grouping_criterion <image/digit/similarity> \
--average_group
```

Running the scripts will produce a parquet file that records per-class Dice performance on every trained model, for each sample in each set. The script automatically evaluates on all set sizes: {1, 2, 4, 6, 8}.

Performance records are saved in format: 
aggregate_type: str, "none" representing baseline, "none" representing ContextFlex 
fusion_type: str, "none" representing baseline, "none" representing ContextFlex
network_size: int, model size in terms of number of parameters 
seed: int, seed number to initialize model weights 
group_id: int, the set ID where the individual image sample belongs 
image_id: int, the image ID of the sample within the set (vary from [0, set_size-1])
group_size: int, set size 
class: int, {0, 1, -1}, where 0 is background, 1 is foreground and -1 is foreground average 
harddice: float, Dice score for each class/ foreground averaged Dice score
