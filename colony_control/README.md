# Controlling the initial state of PSC colonies

We now turn to the PSC stage and built a ML model to analyze how the features from bright-field images at the PSC stage contributed to the final differentiation efficiency. We have extracted 343 features from the bright-field images and computed the Differentiation Efficiency Index from the cTnT images and saved them to `./dataset/CD00-*.pkl`. Please follow the steps below to run the code.

1. Which features are most relevant to high-efficiency differentiation (if the CHIR condition supposed to be optimal)? How is the sample points (wells) distributed in the feature space? How is each feature related to the differentiation efficiency? To answer these question, we build a random forest regression model to derive the feature importance weight (**Fig. 5d**), visualize the feature space spanned by the eight most important feature (**Fig. 5e**), and analyze the relation between each feature with the final efficiency (**Fig. 5f**). Please see [./Feature_importance.ipynb](./Feature_importance.ipynb) for the details.
2. Can ML predicted the final efficiency from the image features at the PSC stage? We applied the random forest regression model for this task, and showed that the predicted efficiency is consistent with the true efficiency (**Fig. 5g**). Please see [./Machine_learning.ipynb](./Machine_learning.ipynb) for the details.



**Note:**

The original image data is too large (~11GB) to be uploaded here, but it will be available upon reasonable request. If you want to reproduce the feature extraction, please email us for the image data `image_data.zip`. The file contains bright-field images (0h, before CHIR treatment), the cell-containing regions, and the final cTnT fluorescence images. Unzip the file and run [./Compute_features.ipynb](./Compute_features.ipynb) (which may take a long time) to obtain the dataset files  `./dataset/CD00-*.pkl`. 

