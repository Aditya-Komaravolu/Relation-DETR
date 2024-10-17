python3 inference.py \
    --image-dir /home/aditya/march22anand_undistorted_images \
    --model-config /home/aditya/Relation-DETR/configs/relation_detr/relation_detr_swin_l_800_1333.py \
    --checkpoint /home/aditya/relation_detr_training_aug20/checkpoints/checkpoint_0/pytorch_model.bin \
    --show-dir /home/aditya/march22anand_undistorted_images_relation_detr_training_aug20_checkpoint_0 \
    --show-conf 0.3
