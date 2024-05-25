#!/bin/bash
# Predict each dataset.

# ########################################## global prarmetters ########################################## #
GPU_ID=0
projectPath='{Your Project Path}'
# MIT1003 path
train_MIT_img_dir=${projectPath}'/datasets/MIT1003/ALLSTIMULI/trainSet'
train_MIT_gt_dir=${projectPath}'/datasets/MIT1003/ALLFIXATIONMAPS/trainSet'
val_MIT_img_dir=${projectPath}'/datasets/MIT1003/ALLSTIMULI/testSet'
val_MIT_gt_dir=${projectPath}'/datasets/MIT1003/ALLFIXATIONMAPS/testSet'
val_MIT_gt_point=${projectPath}'/datasets/MIT1003/ALLFIXATIONLOC/testSet'
# salicon path
train_salicon_img_dir=${projectPath}'/datasets/Salicon/train'
train_salicon_gt_dir=${projectPath}'/datasets/Salicon/Annotations/train'
val_salicon_img_dir=${projectPath}'/datasets/Salicon/val'
val_salicon_gt_dir=${projectPath}'/datasets/Salicon/Annotations/val'
val_salicon_gt_point=${projectPath}'/datasets/Salicon/Annotations/val_point'
# art path
train_art_img_dir=${projectPath}'/datasets/CAT2000/trainSet/Stimuli/Art'
train_art_gt_dir=${projectPath}'/datasets/CAT2000/trainSet/FIXATIONMAPS/Art'
val_art_img_dir=${projectPath}'/datasets/CAT2000/testSet2/Stimuli/Art'
val_art_gt_dir=${projectPath}'/datasets/CAT2000/testSet2/FIXATIONMAPS/Art'
val_art_gt_point=${projectPath}'/datasets/CAT2000/testSet2/FIXATIONLOCS/Art'
# websal path
train_websal_img_dir=${projectPath}'/datasets/WebSal/WebSal/train/image'
train_websal_gt_dir=${projectPath}'/datasets/WebSal/WebSal/train/annotation'
val_websal_img_dir=${projectPath}'/datasets/WebSal/WebSal/val/image'
val_websal_gt_dir=${projectPath}'/datasets/WebSal/WebSal/val/annotation'
# ######################################################################################################## #

# #################################Predict and eval MIT1003 dataset####################################### #
# #predict MIT1003 dataset
#val_img_dir=${val_MIT_img_dir}
#out_dir=${projectPath}'/logs/erfNet_RA_parallel_1layer/MIT1003'
#pretrainedModel=${projectPath}'/checkpoints/ours/MIT1003/erfNet_RA_parallel_1layer_MIT1003_pruned_0.5_final.pt'
#modle=erfNet_RA_parallel_1layer
#current_task=0
#nb_tasks=1
#CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/predict_rap.py \
#   --val_img_dir ${val_img_dir} --out_dir ${out_dir} --pretrainedModel ${pretrainedModel} \
#   --current_task $current_task --nb_tasks $nb_tasks --model ${modle}

# # #eval MIT1003 dataset
#output=${projectPath}'/logs/erfNet_RA_parallel_1layer/MIT1003/erfNet_RA_parallel_1layer_result.txt'
#fixation_folder=${val_MIT_gt_dir}
#point_folder=${val_MIT_gt_point}
#salmap_folder='/home/yzfan3/continual_learning/our_method/logs/MIT1003/salmaps'
#salmap_folder=${out_dir}'/salmaps'
#fxt_loc_name='fixationPts'
#CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/eval_command_rap.py \
#   --output ${output} --fixation_folder ${fixation_folder} --point_folder ${point_folder} \
#   --salmap_folder ${salmap_folder} --fxt_loc_name ${fxt_loc_name}
# ######################################################################################################### #


# #################################Predict and eval salicon dataset####################################### #
# #predict salicon dataset
val_img_dir=${val_salicon_img_dir}
out_dir=${projectPath}'/logs/erfNet_RA_parallel_1layer/salicon/'
pretrainedModel=${projectPath}'/checkpoints/ours/salicon/erfNet_RA_parallel_1layer_salicon_pruned_0.5_final.pt'
modle=erfNet_RA_parallel_1layer
current_task=0
nb_tasks=1
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/predict_rap.py \
   --val_img_dir ${val_img_dir} --out_dir ${out_dir} --pretrainedModel ${pretrainedModel} \
   --current_task $current_task --nb_tasks $nb_tasks --model ${modle}

# #eval salicon dataset
output=${projectPath}'/logs/erfNet_RA_parallel_1layer/salicon/erfNet_RA_parallel_1layer_result.txt'
fixation_folder=${val_salicon_gt_dir}
point_folder=${val_salicon_gt_point}
salmap_folder=${out_dir}'/salmaps'
fxt_loc_name='gaze'
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/eval_command_rap.py \
   --output ${output} --fixation_folder ${fixation_folder} --point_folder ${point_folder} \
   --salmap_folder ${salmap_folder} --fxt_loc_name ${fxt_loc_name}
# ######################################################################################################### #


# ############################Predict and eval finetune model on art dataset############################### #
# #predict art dataset
val_img_dir=${val_art_img_dir}
out_dir=${projectPath}'/logs/erfNet_RA_parallel_1layer/art/finetune'
pretrainedModel=${projectPath}'/checkpoints/ours/art/erfNet_RA_parallel_1layer_art_finetune_0.5.pt'
modle=erfNet_RA_parallel_1layer
current_task=1
nb_tasks=2
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/predict_rap.py \
   --val_img_dir ${val_img_dir} --out_dir ${out_dir} --pretrainedModel ${pretrainedModel} \
   --current_task $current_task --nb_tasks $nb_tasks --model ${modle}

# #eval art dataset.
output=${projectPath}'/logs/erfNet_RA_parallel_1layer/art/finetune/erfNet_RA_parallel_1layer_result.txt'
fixation_folder=${val_art_gt_dir}
point_folder=${val_art_gt_point}
salmap_folder=${out_dir}'/salmaps'
fxt_loc_name='fixLocs'
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/eval_command_rap.py \
   --output ${output} --fixation_folder ${fixation_folder} --point_folder ${point_folder} \
   --salmap_folder ${salmap_folder} --fxt_loc_name ${fxt_loc_name}
# ########################################################################################################### #


# ###############################Predict and eval prune model on art dataset################################# #
# #predict art dataset
val_img_dir=${val_art_img_dir}
out_dir=${projectPath}'/logs/erfNet_RA_parallel_1layer/art/prune'
pretrainedModel=${projectPath}'/checkpoints/ours/art/erfNet_RA_parallel_1layer_art_pruned_0.5_base_0.5_final.pt'
modle=erfNet_RA_parallel_1layer
current_task=1
nb_tasks=2
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/predict_rap.py \
   --val_img_dir ${val_img_dir} --out_dir ${out_dir} --pretrainedModel ${pretrainedModel} \
   --current_task $current_task --nb_tasks $nb_tasks --model ${modle}

# #eval art dataset
output=${projectPath}'/logs/erfNet_RA_parallel_1layer/art/prune/erfNet_RA_parallel_1layer_result.txt'
fixation_folder=${projectPath}'/datasets/CAT2000/testSet2/FIXATIONMAPS/Art'
point_folder=${projectPath}'/datasets/CAT2000/testSet2/FIXATIONLOCS/Art'
salmap_folder=${out_dir}'/salmaps'
fxt_loc_name='fixLocs'
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/eval_command_rap.py \
   --output ${output} --fixation_folder ${fixation_folder} --point_folder ${point_folder} \
   --salmap_folder ${salmap_folder} --fxt_loc_name ${fxt_loc_name}
# ############################################################################################################# #

# #############################Predict and eval finetune model on websal dataset############################### #
# #predict websal dataset
val_img_dir=${val_websal_img_dir}
out_dir=${projectPath}'/logs/erfNet_RA_parallel_1layer/websal'
pretrainedModel=${projectPath}'/checkpoints/ours/websal/erfNet_RA_parallel_1layer_websal_finetune_base_0.5_final.pt'
modle=erfNet_RA_parallel_1layer
current_task=2
nb_tasks=3
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/predict_rap.py \
   --val_img_dir ${val_img_dir} --out_dir ${out_dir} --pretrainedModel ${pretrainedModel} \
   --current_task $current_task --nb_tasks $nb_tasks --model ${modle}

# #eval websal dataset
output=${projectPath}'/logs/erfNet_RA_parallel_1layer/websal/erfNet_RA_parallel_1layer_result.txt'
fixation_folder=${val_websal_img_dir}
point_folder=${val_websal_img_dir}
salmap_folder=${out_dir}'/salmaps'
fxt_loc_name='fixationPts'
CUDA_VISIBLE_DEVICES=$GPU_ID python ${projectPath}/methods/ours/eval_command_rap.py \
   --output ${output} --fixation_folder ${fixation_folder} --point_folder ${point_folder} \
   --salmap_folder ${salmap_folder} --fxt_loc_name ${fxt_loc_name}
# ############################################################################################################### #