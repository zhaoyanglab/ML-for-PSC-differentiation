# Weakly	supervised	learning-based CPC recognition 

Here we provide our custom code for the weakly supervised learning-based CPC recognition at the PSC-CPC stage (stage II). The implementation of ResNeSt and Grad-CAM was borrowed from https://github.com/MachineLP/PyTorch_image_classifier/tree/master/qdnet and https://github.com/jacobgil/pytorch-grad-cam, respectively.

At the training stage, the ResNeSt-101 network learns to classify bright-field image patches as "positive" or "negative". At the inference stage, the trained network is used to localize the target object in the image using Grad-CAM. 

| <img src="./Weakly_Supervised_Learning_Framework.svg" alt="Weakly Supervised Learning Framework" style="zoom:25%;" /> |
| :----------------------------------------------------------: |
| **The framework for weakly-supervised learning-based CPC recognition.**   <br />**a** The training stage. **b** The inference stage. |



## Datasets

Download the datasets from the following link, and unzip it at `./data/`.

* `WeaklySupervisedLearning.zip` (https://drive.google.com/file/d/1qp-k7KTAIALaYPFzaJlX_E8elTVBiOU5/view?usp=share_link), containing 106 paired whole-well bright-field images and hand-labeling masks for training and 35 (with cTnT fluorescence images) for testing. The training and testing images are randomly selected from 6 cell lines.



## Training and testing

Please follow the steps to run the code.

1. Prepare training and test images, run `./Crop_and_Reconstruct/mask_crop.m` to segment full-size brightfield images of day 6 and mask images into patches. Please download the dataset from the following links.
2. Run `./Crop_and_Reconstruct/classification.m` to assign labels (0: negative; 1: positive) to brightfield patches based on mask patches.
3. Copy the paths of the corresponding patches of the training set and test set to `./data/train.csv` and `./data/test.csv`, respectively.
4. Run `./train.py` to train the ResNeSt-101 network.
5. Run `./test.py` to predict the labels of patches in the test set through the trained ResNeSt-101 network.
6. Run `gram_all.py` to get the patch-level Grad-CAM heatmaps and the corresponding binary results.
7. Run `./Crop_and_Reconstruct/splice.m` and `./Crop_and_Reconstruct/spice_BW.m` to reconstruct the full-size Grad-CAM heatmaps and binary prediction of CPC regions from patch-level results.
8. Run `./Evaluation/index_calculate.m` to evaluate the segmentation performance by IoU, accuracy, precision, recall, specificity, and F1 score (**Fig. 3e**); and run `./Evaluation/Pearson_correlation.m` to calculate the Pearson correlation between the predicted proportion of CPC regions and the true differentiation efficiency index (**Fig. 3g**).
