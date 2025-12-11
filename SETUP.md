# Setup
Consider the following instructions for running the model. 

## 1. Download the nnUNet model

Truly, you may follow the instructions from the "Installation Instructions" section of the nnUNet github repository: 

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md

I will repeat them here for the most part however for continuity purposes. If you run into any errors, please troubleshoot with the link above. 

1.1.  If conda is not yet installed on your machine install, and create a conda environment for Python 3.10 or greater. Activate the conda environment.

```
conda create --name [env-name] python=3.10
conda activate [env-name]
```
  
1.2.  Install PyTorch in the conda environment, and ensure that you are installing the proper version given computer configurations. It may be helpful to visit the PyTorch website itself: https://pytorch.org/get-started/locally/
1.3.  In your conda environment, run the command
```
pip install nnunetv2
```

1.4.  Create the following three environment variables in your conda environment. Note that you actually need to create the folders "nnUNet_raw", "nnUNet_preprocessed", and "nnUNet_results" firstThat is, run these commands for Linux/MacOS:
```
export nnUNet_raw="/[path to folder in which you are running the model]/nnUNet_raw"
export nnUNet_preprocessed="/[path to folder in which you are running the model]/nnUNet_preprocessed"
export nnUNet_results="/[path to folder in which you are running the model]/nnUNet_results"

export nnUNet_results="/Users/ethanfazal/test_setup/nnUNet_results"
```

Or these for windows:
```
set nnUNet_raw=C:/Users/fabian/nnUNet_raw
set nnUNet_preprocessed=C:/Users/fabian/nnUNet_preprocessed
set nnUNet_results=C:/Users/fabian/fabian/nnUNet_results
```

## 2. Download the Data
2.1. I have included in the Box Folder my testing dataset, which you may download by downloading the entirety of "Dataset920_HE" from /ML-Histo-Nerve-Segmentation/data/. Note that this includes .tiff files of input images (already downsampled) and their corresponding ground truth labels as .tiff image masks.
2.2. Place the folder you downloaded into the "nnUNet_raw" folder you created earlier. Importantly, do not change anything else about the naming or directory structure.
2.3. Then, navigate to the /ML-Histo-Nerve-Segmentation/models/ folder in Box, and download the folder "Dataset920_HE" from there and place it into the "nnUNet_results" folder you created earlier.

## 3. Run the model
3.1. Run the following command to generate segmentation predictions on the testing dataset. Note that this process may take a bit of time to run for all 20 test images, however it is possible. If you would just like to test on some images, simply remove some from the "imagesTs" folder.

If you have a gpu:
```
nnUNetv2_predict \
    -i $nnUNet_raw/Dataset920_HE/imagesTs \
    -o $nnUNet_raw/Dataset920_HE/preds \
    -d 920 \
    -c 2d \
    -tr nnUNetTrainer \
    -p nnUNetPlans_lowmem \
    -f 0
```

If you would like to use cpu: 
```
nnUNetv2_predict \
    -i $nnUNet_raw/Dataset920_HE/imagesTs \
    -o $nnUNet_raw/Dataset920_HE/preds \
    -d 920 \
    -c 2d \
    -tr nnUNetTrainer \
    -p nnUNetPlans_lowmem \
    -f 0 \
    -device cpu \
    --disable_tta
```

This will output predicted segmentation masks at the filepath "$nnUNet_raw/Dataset920_HE/preds".

## 4. Visualize Results
Consider the following directions to be able to see the results of the model

4.1 Install the visualization software napari with the following command, still within your conda environment. Also install the tifffile library for use handling .tiff files. 
```
conda install -c conda-forge napari pyqt
conda install -c conda-forge tifffile
```
4.2 Download the results-visualizer.py file from the /ML-Histo-Nerve-Segmentation/src/ path in the Box and place it into your original parent directory. Run the program as follows, always with one raw image and its matched predicted segmentation.
```
python3 results-visualizer.py --img "nnUNet_raw/Dataset920_HE/imagesTs/HE_image_001_0000.tiff" --mask "nnUNet_raw/Dataset920_HE/preds/HE_image_001.tiff"
```
4.3 Napari's GUI should open (might take a minute to initialize) but you should then be able to see the model's predicted segmentations overlayed onto the original image. 
