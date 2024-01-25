# Swin-Fetal-Brain-Segmentation

This Swin-UNETR has been developed to automatically segment fetal brain from 3D MRI images and it is based on [this repository](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR). Here you can find:
 
  -  Python code to **Train or fine tune** your Swin-UNETR on your dataset using weight random inizialization or our pre-traind weigth from a rs-fMRI fetal brain segmentation task (reccomanded option);
  - Python code to **Test** our model on new rs-fMRI fetal scans;
  - A folder **logdir** where the output are saved during training;
  - A folder **output** where the output are saved during testing;
  - Images folders: imagesTr and labelsTr (Train and validation images and label), imagesTs and labelsTs (Test images and label)
  -  A folder **weights** which include thers-fMRI fetal brain segmentation task pretrain weights;
  -  A **json** folder which contains two jsons file for training and test;

Before to start, install the necessary dependencies (see below).

# Dependencies
- Conda see [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Please, create a new conda environment with python 3.10.8 and install the requirements.txt file by using **'conda install --file requirements.txt'**.

# Before to start

Download the images and pretrain weights from ...
Change the main path (data_dir) on line 59 of Train.py and line 62 of Test.py

# How to use it

Activate the new conda enviroment with all dependencies and enter 'python path/to/your/pythonfile.py'

# Data

To be upload

# Citation

To be upload

