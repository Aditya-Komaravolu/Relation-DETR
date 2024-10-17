from torch import optim
import sys
sys.path.append("/home/aditya/Relation-DETR")

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict


# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 4    # Increased batch size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.05   # Reduced gradient norm clipping

output_dir = "/home/aditya/relation_detr_training_aug21"  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training


# define dataset for train
coco_path = "/home/aditya/snaglist_sem_aug20"  # /PATH/TO/YOUR/COCODIR
train_dataset = CocoDetection(
    # img_folder=f"{coco_path}/train2017",
    img_folder=f"{coco_path}/train",
    ann_file=f"{coco_path}/annotations/train.json",
    # ann_file=f"{coco_path}/annotations/instances_train2017.json",
    transforms=presets.detr,  # see transforms/presets to choose a transform
    train=True,
)
test_dataset = CocoDetection(
    # img_folder=f"{coco_path}/val2017",
    img_folder=f"{coco_path}/valid",
    # ann_file=f"{coco_path}/annotations/instances_val2017.json",
    ann_file=f"{coco_path}/annotations/valid.json",
    transforms=None,  # the eval_transform is integrated in the model
)

# model config to train
model_path = "configs/relation_detr/relation_detr_swin_l_800_1333.py"

# specify a checkpoint folder to resume, or a pretrained ".pth" to finetune, for example:
# checkpoints/relation_detr_resnet50_800_1333/train/2024-03-22-09_38_50
# checkpoints/relation_detr_resnet50_800_1333/train/2024-03-22-09_38_50/best_ap.pth
resume_from_checkpoint = "/home/aditya/Relation-DETR/relation_detr_swin_l_800_1333_coco_2x.pth"

learning_rate = 1e-4  # initial learning rate
optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-3, betas=(0.9, 0.999))  # Increased weight decay
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(T_max=num_epochs, eta_min=1e-6)  # Cosine Annealing

# This define parameter groups with different learning rate
param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)