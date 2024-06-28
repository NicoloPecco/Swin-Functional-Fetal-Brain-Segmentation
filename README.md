# Swin-Fetal-Brain-Segmentation

This Swin-UNETR has been developed to automatically segment fetal brain from 3D fetal functional MR images and codes are based on [this repository](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR). Here you can find:
 
  -  Python code to **Train or fine tune** your Swin-UNETR on your dataset using weight random inizialization or our pre-traind weigth from a rs-fMRI fetal brain segmentation task (reccomanded option);
  - Python code to **Test** our model on new rs-fMRI fetal scans;
  - A folder **logdir** where the output are saved during training;
  - A folder **output** where the output are saved during testing;
  - Images folders: imagesTr and labelsTr (Train and validation images and label as defined in json file), imagesTs and labelsTs (Test images and label)
  -  A **json** folder which contains two jsons file for training and test;
  -  **Images** and **Swin pretrain weight** can be downloadeed from [here](https://doi.org/10.17632/dyg9dpmgvs.1).

<p align="center">
<img src="https://github.com/NicoloPecco/Swin-Functional-Fetal-Brain-Segmentation/blob/main/Image_results.png" width="1000" height="630">
</p>
<p align="center">
Results of Swin-UNETR model and ground truth with fetal rs-fMRI scan.
</p>

# Before to start

- We reccomand to use Conda - see [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Create a new conda environment with python 3.10.8 and install the pytorch.yml file.
- Download the images and pretrain weights from doi (to be uploaded) and:
   - Place the pretrain weight on the 'weight' folder.
   - If you want to try our images just download them and place it on the correct folder;
   - If you want to use your own images place it on the correct folder and create jsons file with the new images;
- Change the main path (data_dir - 'Insert/your/path') on line 59 of Train.py and line 62 of Test.py

# How to use it

Be sure to work on a visible GPU.

- Open the terminal;
- Activate the new conda enviroment;
- Enter 'python path/to/your/Train.py' or 'python path/to/your/Test.py'

# Citation

Pecco, N., Della Rosa, P. A., Canini, M., Nocera, G., Scifo, P., Cavoretto, P. I., Candiani, M., Falini, A., Castellano, A., & Baldoli, C. (2024). Optimizing performance of transformer-based models for fetal brain MR image segmentation. Radiology: Artificial Intelligence, 0(ja), e230229. https://doi.org/10.1148/ryai.230229

