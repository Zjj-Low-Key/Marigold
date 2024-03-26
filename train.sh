LAUNCH_TRAINING_IMAGE(){

# accelerate config default
# cd .. 
cd training
pretrained_model_name_or_path='/data/local_userdata/zhujiajun/Marigold/checkpoint/Marigold_v1_merged_2'
root_path='/data1/liu'
dataset_name='sceneflow'
trainlist='/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/SceneFlow_With_Occ.list'
vallist='/home/zliu/ECCV2024/Accelerator-Simple-Template/datafiles/sceneflow/FlyingThings3D_Test_With_Occ.list'
output_dir='../outputs'
train_batch_size=2
num_train_epochs=10
gradient_accumulation_steps=4
learning_rate=1e-5
lr_warmup_steps=0
dataloader_num_workers=1
tracker_project_name='sceneflow_pretrain_tracker_img2img'


CUDA_VISIBLE_DEVICES=6,7 accelerate launch --mixed_precision="no"  --multi_gpu depth2image_trainer.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $root_path --vallist $vallist \
                  --output_dir $output_dir \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --gradient_checkpointing \
                  --enable_xformers_memory_efficient_attention \

}


LAUNCH_TRAINING_IMAGE