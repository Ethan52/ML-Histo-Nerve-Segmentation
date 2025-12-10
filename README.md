# ML-Histo-Nerve-Segmentation - Don't grade yet, even if gives 6% penalty will resubmit tonight/tomorrow, very sorry

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

# 3. Video Links



# 4. Evaluation

After training the model, I evaluated how well the model predicts the desired segmentations in the standard manner of evaluation of the DiCE score, as shown in the following tables. This was done specifically using the eval.py script included in the repository. Furthermore, the analysis was done both on the level of each segmentation class, as well as on average, to allow for better interpretation. Consider the mathematical equation of the score to be 

$$
\mathrm{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B| + \epsilon}
$$

for two possible segmentation labels $A$ and $B$, with $\epsilon$ small to avoid division by 0. 

## 4.1 Base Model Analysis

### nnUNet Base Model

| Structure           | Mean Dice | Std Dev  | Min     | Max     |
|--------------------|-----------|----------|---------|---------|
| **Background**      | 0.9850    | 0.0089   | 0.9639  | 0.9974  |
| **Endoneurium**     | 0.9870    | 0.0059   | 0.9662  | 0.9940  |
| **Perineurium**     | 0.9245    | 0.0240   | 0.8769  | 0.9576  |
| **Epineurium**      | 0.9709    | 0.0158   | 0.9312  | 0.9880  |
| **Mean Dice (No Bg)** | 0.9608 | 0.0107   | 0.9433  | 0.9754  |
| **Mean Dice (All)**   | 0.9668 | 0.0093   | 0.9510  | 0.9790  |

Observe that, at least quantitatively, the model appears to be doing very well at predicting the regions of interest desired to be segmented, with a fairly high mean DiCE, and relatively small standard deviation, indicating little variation throughout the predictions, reaffirmed by the min-max intervals also shown above. Notice that the Background, Endoneurium, and Epineurium segmentation classes appear to have been learned slightly better than the Perineurium class. Perhaps to be better informed by the qualitative (image) analysis, however, at a preliminary standpoint, the perineurium is known to be harder to concretely identify, especially defining the boundary between the endoneurium and itself, as well as the epinuerium and itself. Additionally, the perineurium can be much thinner than are the other regions desired to be segmented. Consider also that the standard deviation of the perineurium DiCE scores also indicated the most variation out of all the classes. 

Consider the following overlays depicting the model's predicted segmentation on the left, and the ground truth segmentation on the right, with endoneurium painted red, perineurium blue, and epineurium purple.

<p align="center">
  <img src="002-base-model.png" width="500">
  <img src="002-gt.png" width="500">
</p>

At first glance, as is suggested by the quantitative analysis, the segmentations do not look extremely different, however a deeper dive does reveal more about them. For instance, the image below zooms in to reveal some "holes" in the segmentation of the endoneurium, which should not occur. A possible reason for this is the stark difference in color sometimes seen within the endoneurium, as during the histology process, the dead axons naturally begin to separate from their bundles, revealing some gaps in the endoneurium. However, the ground truth segmentation does not reflect this as something desired to be captured, and thus, this might necessitate a fix in the future. 
<img width="500" alt="image" src="https://github.com/user-attachments/assets/e077fae7-23e2-4d55-a0ff-a913549ca697" />

Moreover, consider the following side-by-side of the model-predicted segmentation (left) and the original histology slide itself (right) zoomed in on the same lower left region of the previous image. Here we see the difficulty in distinguishing the endoneurium from the perineurium for the model. What is slightly interesting is that the model does not segment what I believe to be the entire region of doubt one singular segmentation class but rather "expresses" its uncertainty in its hedging between both classes. 
<p align="center">
    <img width="500" alt="image" src="https://github.com/user-attachments/assets/9d794ff7-ae1a-4d4f-b79f-1ae930230b04" />
    <img width="500" alt="image" src="https://github.com/user-attachments/assets/27696619-fa16-43fa-8f75-b24d5e37a0b6" />
</p>

Additionally, depicted in the same format as above, we notice that the model has difficulty recognizing the presence of entire other nerve structures (perineurium enclosing endoneurium) in very close proximity to the larger perineurium structure. This is likely due to the proximity to the larger perinuerium structure coupled with the similarity in colors, however, this may also be a pitfall of the labeled data, since there is some subtlety in when and in what cases, small nerve structures such as these are treated as deserving of their own perineurium and endoneurium, or if they should simply be lumped together with the epineurium.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/d8096469-5bad-41c0-8f1a-95f41fdd1faa" />
<img width="500" alt="image" src="https://github.com/user-attachments/assets/8dd781ec-2a11-4872-a2f2-88ec57bd9f42" />


Additionally, we have that the model trained with the following metrics shown. Notice that it may appear as though the model is overfitting, the later studies show that this is not truly so, and the model is simply training well, also evidenced by the impressive statistics shown in the beginning.

<p align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/72d987e4-9a56-4714-9b7b-84b465e38025" />
</p>

## 4.1 Ablation Studies

As mentioned, the nnUNet framework utilizes a large amount of data augmentation techniques, one of those being the perturbation of the colors of the image in the training methods. Consider the following table demonstrating the DiCE sccores achieved when training without such techniques, specifically, observe that the scores do not drop that much, if at all. We again see that the endoneurium and epineurium segmentation classes are performing slightly better than the perineurium, however, this too is consistent with our base model. 

### nnUNet - Trained without Color Augmentation (Ablation)

| Structure             | Mean Dice | Std Dev  | Min     | Max     |
|----------------------|-----------|----------|---------|---------|
| **Background**        | 0.9846    | 0.0080   | 0.9669  | 0.9971  |
| **Endoneurium**       | 0.9861    | 0.0093   | 0.9489  | 0.9930  |
| **Perineurium**       | 0.9283    | 0.0201   | 0.8812  | 0.9623  |
| **Epineurium**        | 0.9705    | 0.0157   | 0.9313  | 0.9885  |
| **Mean Dice (No Bg)** | 0.9616    | 0.0089   | 0.9452  | 0.9755  |
| **Mean Dice (All)**   | 0.9674    | 0.0077   | 0.9546  | 0.9789  |

Looking closer at some sample images, consider the side-by-side of the segmentations produced by this trained model (left) and those produced by the original (right). We see pretty much the same effects as before, with the model struggling in some admittedly problematic areas, however, on the whole they look fairly similar. The only interesting factor is that there is not much of the holes in segmentation of the endoneurium as noted with the original model, which might make sense considering the perturbation of colors (or lack thereof) is what is being tested. 

<p align="center">
  <img src="002-color-model.png" width="500">
  <img src="002-base-model.png" width="500">
</p>

What would make this model even more desireable is perhaps a speedup in the training, considering there are less augmentations taking place, however, we do not see this truly, as depicted by the very similar training metrics output.

<p align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/da6ad9a1-96c5-4917-811a-ef94a842d334" />
</p>    

Additionally, I also tested, as an ablation study, whether further downsampling the data had an impact on the performance of the model, as this would have allowed, I thought, better computational efficiency. Considering the reported DiCE scores however, we find that this trained model slightly underperformed in most of the segmentation classes, however not significanty. We again notice the perineurium segmentation class providing the most difficulty, perhaps further exacerbated by the loss of some resolution in training, causing it to be even harder to distinguish a small boundary. 

### nnUNet - Trained with Further Downsampled Data (x8) (Ablation)

| Structure             | Mean Dice | Std Dev  | Min     | Max     |
|----------------------|-----------|----------|---------|---------|
| **Background**        | 0.9710    | 0.0196   | 0.9037  | 0.9965  |
| **Endoneurium**       | 0.9465    | 0.0674   | 0.7060  | 0.9929  |
| **Perineurium**       | 0.8778    | 0.0481   | 0.7329  | 0.9585  |
| **Epineurium**        | 0.9440    | 0.0309   | 0.8680  | 0.9807  |
| **Mean Dice (No Bg)** | 0.9228    | 0.0411   | 0.7851  | 0.9617  |
| **Mean Dice (All)**   | 0.9348    | 0.0350   | 0.8148  | 0.9693  |

Consider also the comparison of segmentation outputs as compared between each of the models. From a wide view, it is evident that there are issues with the segmentation classes, specifically noting the holes apparent in epineurium class, as well as considerable spilling over of the perineurium segmentation class into the endoneurium and epineurium. On the whole, it seems that the downsampling by a further factor of 2 was certainly felt by the model and was thus not able to perform as well as other model configurations have done.    

<p align="center">
  <img src="013-down-model.png" width="500">
  <img src="013-base-model.png" width="500">
</p>

### nnUNet - Trained with Early Stopping

| Structure             | Mean Dice | Std Dev  | Min     | Max     |
|----------------------|-----------|----------|---------|---------|
| **Background**        | 0.9758    | 0.0097   | 0.9535  | 0.9943  |
| **Endoneurium**       | 0.9672    | 0.0161   | 0.9166  | 0.9839  |
| **Perineurium**       | 0.8225    | 0.0544   | 0.6841  | 0.9226  |
| **Epineurium**        | 0.9491    | 0.0197   | 0.8977  | 0.9758  |
| **Mean Dice (No Bg)** | 0.9129    | 0.0236   | 0.8649  | 0.9532  |
| **Mean Dice (All)**   | 0.9286    | 0.0184   | 0.8930  | 0.9592  |

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
