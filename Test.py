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
    LoadImage,
    EnsureChannelFirstd,
    Invertd,
    AsDiscreted,
    SaveImaged,
)

from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
from monai.handlers.utils import from_engine

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
    DataLoader,
    NiftiSaver,
)

import numpy as np
import torch

print_config()
loader = LoadImage()

model = SwinUNETR(
    img_size=(64, 64, 64),
    in_channels=1,
    out_channels=2,
    feature_size=48,
    use_checkpoint=True,
)

data_dir = "/beegfs/scratch/ric.dellarosa/pecco.nicolo/REVISIONS/Test_github/"
logdir= os.path.normpath(os.path.join(data_dir,'logdir'))
json_path = os.path.normpath(os.path.join(data_dir,'json/Test_set.json'))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Need minimal transforms just to be able to show the unmodified originals

datalist = load_decathlon_datalist(base_dir=data_dir,
                                       data_list_file_path=json_path,
                                       is_segmentation=True,
                                       data_list_key="test")

val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(2.25,2.25,3), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
        ),
    ]
)

val_org_ds = CacheDataset(
        data=datalist,
        transform=val_org_transforms,
        cache_num=6,
        cache_rate=1.0,
        num_workers=10
        )

val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=10)

post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=val_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device=device,
    ),
    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    AsDiscreted(keys="label", to_onehot=2),
])
 
    
    
post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=val_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=os.path.normpath(os.path.join(data_dir,'out/')), output_postfix="seg", resample=True,mode='nearest'),
])

path_to_model=os.path.normpath(os.path.join(logdir,'best_metric_model.pth'))
model.load_state_dict(torch.load(path_to_model))
model.cuda()
model.eval()

dice_metric = DiceMetric(include_background=False, reduction="mean")
HD_metric=HausdorffDistanceMetric(include_background=False, reduction="mean")
dice_vals = list()
HD_vals=list()

with torch.no_grad():
    for val_data in val_org_loader:
        current_name=val_data["label"].meta["filename_or_obj"]
        val_inputs = val_data["image"].to(device)
        val_data["label"]=val_data["label"].to(device)
#        val_label=val_data["label"].to(device)
#        print(val_data["label"].to(device))
        roi_size = (64,64,64)
        sw_batch_size = 1
        val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
        print(current_name)
        str1 = ""
        for ele in current_name:
            str1 += ele
        
        sep=str1.rsplit(".",1)
        name=sep[0]+'screen.png'
        name1=sep[0]+'screen1.png'

        #plt.figure("check", (15, 9))
        #plt.subplot(2, 3, 1)
        #plt.title("Slice 7")
        #plt.imshow(torch.argmax(val_data["pred"] , dim=1).detach().cpu()[0, :, :, 7]) 
        
        #plt.subplot(2, 3,2)
        #plt.title("Slice 10")
        #plt.imshow(torch.argmax(val_data["pred"] , dim=1).detach().cpu()[0, :, :, 10]) 
        
        #plt.subplot(2, 3, 3)
        #plt.title("Slice 13")
        #plt.imshow(torch.argmax(val_data["pred"] , dim=1).detach().cpu()[0, :, :, 13]) 
        
        #plt.subplot(2, 3, 4)
        #plt.title("Slice 17")
        #plt.imshow(torch.argmax(val_data["pred"] , dim=1).detach().cpu()[0, :, :, 17]) 
        
        #plt.subplot(2, 3, 5)
        #plt.title("Slice 20")
        #plt.imshow(torch.argmax(val_data["pred"] , dim=1).detach().cpu()[0, :, :, 20]) 
        
        #plt.subplot(2, 3, 6)
        #plt.title("Slice 23")
        #plt.imshow(torch.argmax(val_data["pred"] , dim=1).detach().cpu()[0, :, :, 23]) 
        #plt.savefig(os.path.join(name1),dpi=100)
        
        temp=val_data
        
        print('#########################################################################')
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice = dice_metric.aggregate().item()
        dice_vals.append(dice)
        #val_sq=val_outputs
        #y_pred.squeeze(1)
        HD_metric(y_pred=val_outputs, y=val_labels)
        HD = HD_metric.aggregate().item()
        HD_vals.append(HD)
        print(dice)
        
        temp["pred"]=1-temp["pred"]
        
        temp = [post_transforms(i) for i in decollate_batch(temp)]
             
os.chdir(logdir)        
with open('Dice_test.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ')
    my_writer.writerow(dice_vals) 
with open('bAHD_test.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ')
    my_writer.writerow(HD_vals)
