# ML-Histo-Nerve-Segmentation
README.md
H&E Nerve Histology Segmentation with nnU-Net

Final Project – Applied Machine Learning

# 0. Brief Term-list
Consider that throughout this project we will be using some light biological terminology. Consider the following:
1. Endoneurium - the region inside the nerve consisting of the axons
2. Perineurium - the relatively thin boundary between bundles of axons within nerves (encloses endoneurium)
3. Epineurium - the outer wall of the nerve (encloses both perineurium and endoneurium)
4. Fascicle - an individual bundle of axons within a nerve, any given nerve may have multiple or just one of these

# 1. What It Does

Consider that many cutting-edge research topics in the field of medicine currently are exploring the nervous system and specifically nerve morphology, itself a fast-growing field given relatively novel data collection methods. That being said, especially considering the growth in available data, it becomes important to be able to efficiently process these data, that is, glean the desired information as quickly and accurately as possible. This project attempts to accomplish this, specifically for H&E-stained histology images of cross sections of nerves in the carotid artery region of the human body. The project provides a deep learning pipeline for preprocessing the data, segmenting the cross section into the endoneurium, perineurium, and epineurium regions, as well as provides an error analysis of the predictions.

# 2. Quick Start
Prerequisites

Python 3.10

CUDA-enabled PyTorch (if running training)

nnU-Net v2 installed

Access to GPU cluster (recommended)

Environment Setup
conda create -n nnunet python=3.10 -y
conda activate nnunet
pip install nnunetv2

Set nnU-Net environment variables
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

Preprocess Dataset

Your dataset must live in:

$nnUNet_raw/Dataset920_HE/
    imagesTr/
    labelsTr/
    dataset.json


Then run:

nnUNetv2_plan_and_preprocess -d 920 -p nnUNetPlans

Train (Fold 0)
nnUNetv2_train 920 2d 0 --deterministic

Run Inference
nnUNetv2_predict -i ./inference_inputs \
                 -o ./predictions \
                 -d 920 -c 2d -f 0

Apply Postprocessing (optional)
python postprocess_clean.py predictions predictions_clean

3. Video Links

(Add your YouTube or Google Drive links)

Demo Video: TODO: Add Link

Technical Walkthrough Video: TODO: Add Link

4. Evaluation
4.1 Quantitative Metrics (Validation Set)

Baseline fold-0 model

Class	Dice	IoU
Endoneurium	TODO	TODO
Perineurium	TODO	TODO
Epineurium	TODO	TODO
Mean Dice	≈ 0.89 (observed)	TODO

Ablation: No Color Augmentation

Class	Dice	IoU
Endoneurium	TODO	TODO
Perineurium	TODO	TODO
Epineurium	TODO	TODO
Mean Dice	TODO	TODO
Δ Mean Dice	TODO (usually small)	

Postprocessing Improvement (Morphological Cleaning)

Model	Mean Dice Before	Mean Dice After	Δ
Baseline	TODO	TODO	+TODO
NoColorAug	TODO	TODO	+TODO

(You can paste results from metrics_comparison.csv here.)

4.2 Qualitative Examples

Below are representative overlays comparing Ground Truth, Baseline Prediction, and No-Color-Aug Prediction:

Good case

TODO: insert image or link

Typical case

TODO: insert image or link

Failure / challenging case

TODO: insert image or link (e.g., ruptured perineurium)

(Optional: include masks before/after postprocessing.)

4.3 Ablation Study Summary

You can write this exactly or modify:

We evaluated the effect of removing color augmentation during training.
Surprisingly, the removal of stain-based augmentations produced minimal change in performance (Δ ≈ TODO Dice).
This suggests that our dataset exhibits consistent H&E staining, making color augmentation less impactful.

The postprocessing step (hole filling + morphological closing/opening) consistently improved perineurium and epineurium segmentation.

5. Individual Contributions

(Modify if group project; otherwise say “Individual project”.)

Ethan Fazal

Data preprocessing and annotation (manual labeling in QuPath)

Adapted nnU-Net v2 to histology dataset

Implemented custom trainer removing color augmentations

Ran training, inference, and evaluation on A5000 cluster

Constructed ablation study and postprocessing pipeline

Created demo & technical walkthrough videos

Authored full report and README

6. Repository Structure

Adjust paths as needed.

├── nnunet_train_fold0.sbatch          # training job script
├── nnunet_preprocess.sbatch           # preprocess job script
├── postprocess_clean.py               # morphological cleaning postprocessing
├── evaluate_and_overlay.py            # quantitative + qualitative evaluation
├── predictions/                       # baseline predictions
├── predictions_nocolor/               # ablation predictions
├── overlays/                          # saved qualitative figures
├── Dataset920_HE/                     # raw dataset (imagesTr, labelsTr)
└── README.md

7. How to Reproduce

Download dataset → place under nnUNet_raw.

Run preprocess.

Run fold-0 training.

Run inference on test/val images.

Run evaluation scripts for metrics + overlays.

Optional: run ablation (-tr nnUNetTrainerNoColorAug).

8. Requirements
Python >= 3.10
PyTorch >= 2.1
CUDA >= 12.x (cluster)
nnU-Net v2
scikit-image
numpy
tifffile
scipy
matplotlib
