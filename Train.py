import os
import csv
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    AddChanneld,
    ToTensord,
    DivisiblePadd,
)

from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
    DataLoader,
)

import numpy as np
import torch

print_config()


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096*4, rlimit[1]))

def main():

    # Define images and window sizeparameters 
    cropsize=(64,64,64)
    px_dim=(2.25,2.25,3)
    
    # Define paths for running the script
    data_dir = "/beegfs/scratch/ric.dellarosa/pecco.nicolo/REVISIONS/Test_github/"
    pretrained_path = os.path.normpath(os.path.join(data_dir, 'weight/model_swinvit.pt'))
    logdir= os.path.normpath(os.path.join(data_dir,'logdir'))
    json_path = os.path.normpath(os.path.join(data_dir,'json/Train100.json'))
    pretrained_path_fetal=os.path.normpath(os.path.join(data_dir, 'weight/Fetal_pretrain.pth'))

    # Training Hyper-parameters
    lr = 1e-4
    max_iterations = 30000
    eval_num = 100

    if os.path.exists(logdir)==False:
        os.mkdir(logdir)

    # Training & Validation Transform chain
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=px_dim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cropsize,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
            DivisiblePadd(keys=["image", "label"],k=64),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=px_dim,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(base_dir=data_dir,
                                       data_list_file_path=json_path,
                                       is_segmentation=True,
                                       data_list_key="training")

    val_files = load_decathlon_datalist(base_dir=data_dir,
                                        data_list_file_path=json_path,
                                        is_segmentation=True,
                                        data_list_key="validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=60,
        cache_rate=1.0,
        num_workers=10,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=10, pin_memory=False,persistent_workers=False
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_num=15,
        cache_rate=1.0,
        num_workers=10
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=10, pin_memory=False,persistent_workers=False
    )

    case_num = 0
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    img_shape = img.shape
    label_shape = label.shape
    #print(f"image shape: {img_shape}, label shape: {label_shape}")
#    print(val_ds[case_num])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=cropsize,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
    )
    
    #To use original weithts (No Fetal)
    #weight = torch.load(pretrained_path)
    #model.load_from(weights=weight)
    #model.to(device)

    #To use pretrain fetal weigth
    model.load_state_dict(torch.load(pretrained_path_fetal))
    model.cuda()
    print("Using pretrained self-supervied Swin UNETR backbone weights!")

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    HD_metric=HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    metric_values_HD=[]
    all_dice_fold = []
    metric_values_HD=[]
    HD_metric=HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        hasudorf_vals= list()
        
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs,cropsize, 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                HD_metric(y_pred=val_output_convert, y=val_labels_convert)
                HD = HD_metric.aggregate().item()
                hasudorf_vals.append(HD)
                
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
                )

            dice_metric.reset()
            HD_metric.reset()

        mean_dice_val = np.mean(dice_vals)
        mean_HD_val = np.mean(hasudorf_vals)
        
        return mean_dice_val, mean_HD_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                 #print(x.shape)
                 model.eval()  
                 logit_map = model(x)
                 loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
            )

            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )

                dice_val, HD_val = validation(epoch_iterator_val)

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                metric_values_HD.append(HD_val)
                
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(logdir, "best_metric_model.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                    
                plt.figure(1, (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Iteration Average Loss")
                x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.grid()
                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [eval_num * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.grid()
                plt.savefig(os.path.join(logdir, 'Quick_update.png'))
                plt.clf()
                plt.close(1)

            global_step += 1
            
        return global_step, dice_val_best, global_step_best

    print(metric_values)
    
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    model.load_state_dict(torch.load(os.path.join(logdir, "best_metric_model.pth")))
    os.chdir(logdir)
    with open('Dice_score_train.csv', 'w', newline = '') as csvfile:
    	my_writer = csv.writer(csvfile, delimiter = ' ')
    	my_writer.writerow(metric_values)  
    print(
        f"train completed, best_metric: {dice_val_best:.4f} "
        f"at iteration: {global_step_best}"
    )
    with open('bAHD_score_train.csv', 'w', newline = '') as csvfile:
    	my_writer = csv.writer(csvfile, delimiter = ' ')
    	my_writer.writerow(metric_values_HD) 

if __name__=="__main__":
    main()
