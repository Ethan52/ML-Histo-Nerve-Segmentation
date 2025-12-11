# Attribution and Acknowledgments

This project builds upon several open-source tools, libraries, and methodologies. We gratefully acknowledge the following:

## Primary Framework

### nnU-Net (no new U-Net)
This project is built on **nnU-Net v2**, a self-configuring framework for medical image segmentation.

**Repository:** https://github.com/MIC-DKFZ/nnUNet  
**License:** Apache License 2.0  
**Authors:** Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein  
**Institution:** German Cancer Research Center (DKFZ), Heidelberg, Germany

**Key Contributions from nnU-Net:**
- Automated architecture configuration based on dataset properties
- Preprocessing pipeline and patch-based training strategy
- Training loop, loss functions, and optimization schedule
- Evaluation metrics and cross-validation framework
- The core U-Net architecture implementation

## Deep Learning Framework

### PyTorch

**Repository:** https://github.com/pytorch/pytorch  
**License:** BSD-3-Clause License

## Data Augmentation

### batchgenerators
Data augmentation pipeline used by nnU-Net.

**Repository:** https://github.com/MIC-DKFZ/batchgenerators  
**License:** Apache License 2.0  
**Authors:** Division of Medical Image Computing, German Cancer Research Center (DKFZ)

**Key Features Used:**
- Spatial transformations (rotation, scaling, elastic deformation)
- Color augmentations (brightness, contrast, gamma)
- Noise augmentations (Gaussian noise and blur)

## U-Net Architecture

### Original U-Net
The foundational architecture that nnU-Net is based on.

## Institutional Support

- **Computing Resources:** Grill Lab Partition
- **Cluster:** Duke Compute Cluseter

## Dataset

Case Western Reserve University BME (in relation with Grill Lab)

## Acknowledgment of Assistance

Obtained some samples of labeled data from Grill Lab, note that the majority were still hand-produced for the purpose of this project. 

## AI Acknowledgement

Note that ChatGPT was used in the production of this project, especially for debugging of errors and learning how to handle working with a compute cluster. 

**Contact:** ethan.fazal@duke.edu
