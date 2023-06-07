# Transferring the image-based strategy to nephric differentiation

The success of the image-based strategy in cardiac differentiation encouraged us to transfer our strategy to nephric differentiation. Here we aim to assess the CHIR concentration on day 4 using machine learning. We again model the problem as an image classification problem, where day 4 bright-field image patches are expected to be classified as "low", "optimal" or "high". Concretely, we adopt the logistic regression classifier built on top of SIFT local features.



Please follow the steps below to reproduce the results in our paper.

1. Download the dataset of day 4 bright-field image patches of nephric differentiation from https://drive.google.com/file/d/1EutGhtieLbAPCYnDxCfE-mYKBtYOaJtX/view?usp=sharing and unzip it. The dataset contains 4855 non-overlapping image patches labeled with "low", "optimal", or "high". They should be saved as `./image_patches/(low|optimal|high)/S*/patch*.png`.
2. Run [SIFT_feature_extraction.ipynb](./SIFT_feature_extraction.ipynb) to divide the dataset into a training set and a test set, initialize the SIFT feature extractor. The information about the training set and test set are saved as `dataset_train.pkl` and `dataset_test.pkl`. The fitted SIFT feature extractor is saved as `feature_extractor.pkl`. The code also computes the feature vectors of the training set and visualize the feature space using t-SNE (**Supplementary Fig. S15d**). 
3. Run [classification.ipynb](./classification.ipynb) to train and test a logistic regression model for image patch classification (**Supplementary Fig. S15e,f**).