# ML-Histo-Nerve-Segmentation
README.md
H&E Nerve Histology Segmentation with nnUNet

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

# 4. Evaluation
## 4.1 Quantitative Metrics (Testing Set)

### nnUNet Base Model

| Structure           | Mean Dice | Std Dev  | Min     | Max     |
|--------------------|-----------|----------|---------|---------|
| **Background**      | 0.9850    | 0.0089   | 0.9639  | 0.9974  |
| **Endoneurium**     | 0.9870    | 0.0059   | 0.9662  | 0.9940  |
| **Perineurium**     | 0.9245    | 0.0240   | 0.8769  | 0.9576  |
| **Epineurium**      | 0.9709    | 0.0158   | 0.9312  | 0.9880  |
| **Mean Dice (No Bg)** | 0.9608 | 0.0107   | 0.9433  | 0.9754  |
| **Mean Dice (All)**   | 0.9668 | 0.0093   | 0.9510  | 0.9790  |


### nnUNet - Trained without Color Augmentation (Ablation)

| Structure             | Mean Dice | Std Dev  | Min     | Max     |
|----------------------|-----------|----------|---------|---------|
| **Background**        | 0.9846    | 0.0080   | 0.9669  | 0.9971  |
| **Endoneurium**       | 0.9861    | 0.0093   | 0.9489  | 0.9930  |
| **Perineurium**       | 0.9283    | 0.0201   | 0.8812  | 0.9623  |
| **Epineurium**        | 0.9705    | 0.0157   | 0.9313  | 0.9885  |
| **Mean Dice (No Bg)** | 0.9616    | 0.0089   | 0.9452  | 0.9755  |
| **Mean Dice (All)**   | 0.9674    | 0.0077   | 0.9546  | 0.9789  |

### nnUNet - Trained with Further Downsampled Data (x8) (Ablation)

| Structure             | Mean Dice | Std Dev  | Min     | Max     |
|----------------------|-----------|----------|---------|---------|
| **Background**        | 0.9710    | 0.0196   | 0.9037  | 0.9965  |
| **Endoneurium**       | 0.9465    | 0.0674   | 0.7060  | 0.9929  |
| **Perineurium**       | 0.8778    | 0.0481   | 0.7329  | 0.9585  |
| **Epineurium**        | 0.9440    | 0.0309   | 0.8680  | 0.9807  |
| **Mean Dice (No Bg)** | 0.9228    | 0.0411   | 0.7851  | 0.9617  |
| **Mean Dice (All)**   | 0.9348    | 0.0350   | 0.8148  | 0.9693  |

### nnUNet - Trained with Early Stopping

| Structure             | Mean Dice | Std Dev  | Min     | Max     |
|----------------------|-----------|----------|---------|---------|
| **Background**        | 0.9758    | 0.0097   | 0.9535  | 0.9943  |
| **Endoneurium**       | 0.9672    | 0.0161   | 0.9166  | 0.9839  |
| **Perineurium**       | 0.8225    | 0.0544   | 0.6841  | 0.9226  |
| **Epineurium**        | 0.9491    | 0.0197   | 0.8977  | 0.9758  |
| **Mean Dice (No Bg)** | 0.9129    | 0.0236   | 0.8649  | 0.9532  |
| **Mean Dice (All)**   | 0.9286    | 0.0184   | 0.8930  | 0.9592  |

## 4.2 Qualitative Examples

Below are representative overlays for each of Ground Truth, Baseline Prediction, and No-Color-Aug Prediction:

Good case

TODO: 

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
