#!/bin/bash
# Adds tasks in sequence using the iterative pruning + re-training method.

# ################################ global prarmetters ################################ #
GPU_ID=0
projectPath='{Your Project Path}'
# MIT1003 path
train_MIT_img_dir=${projectPath}'/datasets/MIT1003/ALLSTIMULI/trainSet'
train_MIT_gt_dir=${projectPath}'/datasets/MIT1003/ALLFIXATIONMAPS/trainSet'
val_MIT_img_dir=${projectPath}'/datasets/MIT1003/ALLSTIMULI/testSet'
val_MIT_gt_dir=${projectPath}'/datasets/MIT1003/ALLFIXATIONMAPS/testSet'
# salicon path
train_salicon_img_dir=${projectPath}'/datasets/Salicon/train'
train_salicon_gt_dir=${projectPath}'/datasets/Salicon/Annotations/train'
val_salicon_img_dir=${projectPath}'/datasets/Salicon/val'
val_salicon_gt_dir=${projectPath}'/datasets/Salicon/Annotations/val'
# art path
train_art_img_dir=${projectPath}'/datasets/CAT2000/trainSet/Stimuli/Art'
train_art_gt_dir=${projectPath}'/datasets/CAT2000/trainSet/FIXATIONMAPS/Art'
val_art_img_dir=${projectPath}'/datasets/CAT2000/testSet2/Stimuli/Art'
val_art_gt_dir=${projectPath}'/datasets/CAT2000/testSet2/FIXATIONMAPS/Art'
# websal path
train_websal_img_dir=${projectPath}'/datasets/WebSal/WebSal/train/image'
train_websal_gt_dir=${projectPath}'/datasets/WebSal/WebSal/train/annotation'
val_websal_img_dir=${projectPath}'/datasets/WebSal/WebSal/val/image'
val_websal_gt_dir=${projectPath}'/datasets/WebSal/WebSal/val/annotation'
# ####################################################################################### #

# ############################# pretrain on salicon/MIT1003 ############################# #
epochs=150
batch_size=16
model=erfNet_RA_parallel_1layer # erfNet_RA_parallel_1layer / erfNet_RA_parallel_2layer025
dataset='salicon'
pretrainedModel=${projectPath}'/model_hub/refnet/erfnet_encoder_pretrained.pth.tar'
savedir=${projectPath}'/checkpoints/pretrain/salicon_erfnet_RA_parallel_2'
current_task=0
nb_tasks=1

CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/train_new_task_step1.py \
--epochs ${epochs} --batch_size ${batch_size} --model ${model} --dataset ${dataset} \
--pretrainedModel ${pretrainedModel} --savedir ${savedir} \
--current_task ${current_task} --nb_tasks ${nb_tasks} \
--train_MIT_img_dir ${train_MIT_img_dir} --train_MIT_gt_dir ${train_MIT_gt_dir} --val_MIT_img_dir ${val_MIT_img_dir} --val_MIT_gt_dir ${val_MIT_gt_dir} \
--train_salicon_img_dir ${train_salicon_img_dir} --train_salicon_gt_dir ${train_salicon_gt_dir} --val_salicon_img_dir ${val_salicon_img_dir} --val_salicon_gt_dir ${val_salicon_gt_dir} \
--train_art_img_dir ${train_art_img_dir} --train_art_gt_dir ${train_art_gt_dir} --val_art_img_dir ${val_art_img_dir} --val_art_gt_dir ${val_art_gt_dir} \
--train_websal_img_dir ${train_websal_img_dir} --train_websal_gt_dir ${train_websal_gt_dir} --val_websal_img_dir ${val_websal_img_dir} --val_websal_gt_dir ${val_websal_gt_dir}
# ####################################################################################### #

# ####################### init_dump, preprocess pretrained model ####################### #
arch=erfNet_RA_parallel_1layer
init_dump=True
pretrainedModel=${projectPath}'checkpoints/pretrain/salicon_erfnet_RA_parallel_2/model_best_salicon_erfNet_RA_parallel_1layer_150_16RAPFT.pth'
init_savedir=${projectPath}'/checkpoints/ours/salicon/'
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/train_salicon_1.py \
 --arch $arch --pretrainedModel $pretrainedModel --init_dump --init_savedir $init_savedir
# ####################################################################################### #

# ############################### Prune on salicon dataset ############################## #
dataset='salicon'
arch=erfNet_RA_parallel_1layer
mode=prune
finetune_layers='all'
lr=1e-4
lr_decay_every=5
lr_decay_factor=0.1
batch_size=16
pruned_savename=${projectPath}'/checkpoints/ours/salicon/'
loadname=${projectPath}'/checkpoints/ours/salicon/erfNet_RA_parallel_1layer_salicon.pt'
prune_perc_per_layer=0.5
post_prune_epochs=150

CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/train_salicon_1.py \
--dataset ${dataset} --arch $arch --mode $mode --finetune_layers ${finetune_layers} \
--lr ${lr} --lr_decay_every ${lr_decay_every} --lr_decay_factor ${lr_decay_factor} \
--batch_size ${batch_size} --save_prefix ${pruned_savename} --loadname ${loadname} \
--prune_perc_per_layer ${prune_perc_per_layer} --post_prune_epochs ${post_prune_epochs} \
--train_MIT_img_dir ${train_MIT_img_dir} --train_MIT_gt_dir ${train_MIT_gt_dir} --val_MIT_img_dir ${val_MIT_img_dir} --val_MIT_gt_dir ${val_MIT_gt_dir} \
--train_salicon_img_dir ${train_salicon_img_dir} --train_salicon_gt_dir ${train_salicon_gt_dir} --val_salicon_img_dir ${val_salicon_img_dir} --val_salicon_gt_dir ${val_salicon_gt_dir}
# ####################################################################################### #

# ############################### Finetune on art dataset ############################### #
dataset='art'
dataset_mask='salicon'
arch=erfNet_RA_parallel_1layer
mode=finetune
finetune_layers='all'
lr=5e-4
lr_decay_every=20
lr_decay_factor=0.1
finetune_epochs=500
batch_size=12
save_prefix=${projectPath}'/checkpoints/ours/art/'
loadname=${projectPath}'/checkpoints/ours/salicon/erfNet_RA_parallel_1layer_salicon_pruned_0.5_final.pt'
prune_perc_per_layer=0.5
post_prune_epochs=2
current_task=1
nb_tasks=2

CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/train_art_2.py \
--dataset ${dataset} --dataset_mask ${dataset_mask} --arch $arch --mode $mode --finetune_layers ${finetune_layers} \
--lr ${lr} --lr_decay_every ${lr_decay_every} --lr_decay_factor ${lr_decay_factor} \
--finetune_epochs ${finetune_epochs} --batch_size ${batch_size} --save_prefix ${save_prefix} --loadname ${loadname} \
--prune_perc_per_layer ${prune_perc_per_layer} --post_prune_epochs ${post_prune_epochs} \
--current_task=${current_task} --nb_tasks ${nb_tasks} \
--train_art_img_dir ${train_art_img_dir} --train_art_gt_dir ${train_art_gt_dir} --val_art_img_dir ${val_art_img_dir} --val_art_gt_dir ${val_art_gt_dir}
# ####################################################################################### #

# ################################ prune on art dataset ################################# #
dataset='art'
dataset_mask='art'
arch=erfNet_RA_parallel_1layer
mode=prune
finetune_layers='all'
lr=5e-4
lr_decay_every=10
lr_decay_factor=0.1
batch_size=12
save_prefix=${projectPath}'/checkpoints/ours/art/'
loadname=${projectPath}'/checkpoints/ours/art/erfNet_RA_parallel_1layer_art_finetune_0.5.pt'
prune_perc_per_layer=0.5
post_prune_epochs=50
current_task=1
nb_tasks=2

CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/train_art_2.py \
--dataset ${dataset} --dataset_mask ${dataset_mask} --arch $arch --mode $mode --finetune_layers ${finetune_layers} \
--lr ${lr} --lr_decay_every ${lr_decay_every} --lr_decay_factor ${lr_decay_factor} \
--batch_size ${batch_size} --save_prefix ${save_prefix} --loadname ${loadname} \
--prune_perc_per_layer ${prune_perc_per_layer} --post_prune_epochs ${post_prune_epochs} \
--current_task=${current_task} --nb_tasks ${nb_tasks} \
--train_art_img_dir ${train_art_img_dir} --train_art_gt_dir ${train_art_gt_dir} --val_art_img_dir ${val_art_img_dir} --val_art_gt_dir ${val_art_gt_dir}
# ####################################################################################### #

# ############################# Finetune on websal dataset ############################## #
dataset='websal'
dataset_mask='art'
arch=erfNet_RA_parallel_1layer
mode=finetune
finetune_layers='all'
lr=5e-4
lr_decay_every=20
lr_decay_factor=0.1
finetune_epochs=500
batch_size=6
save_prefix=${projectPath}'/checkpoints/ours/websal/'
loadname=${projectPath}'/checkpoints/ours/art/erfNet_RA_parallel_1layer_art_pruned_0.5_base_0.5_final.pt'
prune_perc_per_layer=0.5
post_prune_epochs=2
current_task=2
nb_tasks=3

CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/train_websal_3.py \
--dataset ${dataset} --dataset_mask ${dataset_mask} --arch $arch --mode $mode --finetune_layers ${finetune_layers} \
--lr ${lr} --lr_decay_every ${lr_decay_every} --lr_decay_factor ${lr_decay_factor} \
--finetune_epochs ${finetune_epochs} --batch_size ${batch_size} --save_prefix ${save_prefix} --loadname ${loadname} \
--prune_perc_per_layer ${prune_perc_per_layer} --post_prune_epochs ${post_prune_epochs} \
--current_task=${current_task} --nb_tasks ${nb_tasks} \
--train_art_img_dir ${train_art_img_dir} --train_art_gt_dir ${train_art_gt_dir} --val_art_img_dir ${val_art_img_dir} --val_art_gt_dir ${val_art_gt_dir} \
--train_websal_img_dir ${train_websal_img_dir} --train_websal_gt_dir ${train_websal_gt_dir} --val_websal_img_dir ${val_websal_img_dir} --val_websal_gt_dir ${val_websal_gt_dir}
# ####################################################################################### #