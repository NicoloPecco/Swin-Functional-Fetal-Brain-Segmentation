# Swin-Fetal-Brain-Segmentation

This Swin-UNETR has been developed to automatically segment fetal brain from 3D fetal functional MR images and it is based on [this repository](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR). Here you can find:
 
  -  Python code to **Train or fine tune** your Swin-UNETR on your dataset using weight random inizialization or our pre-traind weigth from a rs-fMRI fetal brain segmentation task (reccomanded option);
  - Python code to **Test** our model on new rs-fMRI fetal scans;
  - A folder **logdir** where the output are saved during training;
  - A folder **output** where the output are saved during testing;
  - Images folders: imagesTr and labelsTr (Train and validation images and label as defined in json file), imagesTs and labelsTs (Test images and label)
  -  A **json** folder which contains two jsons file for training and test;
  -  **Images** and **pretrain weight** can be downloadeed from **doi (to be uploaded)**.

Before to start, install the necessary dependencies (see below).

# Dependencies
- We reccomand to use Conda - see [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Create a new conda environment with python 3.10.8 and install the requirements.txt file by using **'conda install --file requirements.txt'**.

# Before to start

- Download the images and pretrain weights from doi (to be uploaded);
- Change the main path (data_dir - Insert/your/path) on line 59 of Train.py and line 62 of Test.py

# How to use it

Be sure to work on a visible GPU.

- Open the terminal;
- Activate the new conda enviroment;
- Enter 'python path/to/your/Train.py' or 'python path/to/your/Test.py'

# Citation

To be upload

