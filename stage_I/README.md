# Machine learning-based CHIR dose assessment and adjustment

Here we provide our code for the machine-learning based CHIR dose assessment and adjustment at stage I. We adopted the logistic regression model on hand-crafted features of 0-12h bright-field image streams.



## Dataset preparation & feature extraction

The image stream data are saved as `./data/image/CD01-*/S*/T*.png`, where `CD01-*` (`CD01-1`, `CD01-2`, `CD01-3`, `CD01-4`) is the batch name, `S*` (`S1`, `S2`, ..., `S96`) is the index of well, and `T*.png ` (`T1.png`, `T2.png`, ..., `T10.png`) is the preprocessed bright-field image of the well at different time step of the image stream. 

Then, run the following command

```shell
cd ./data
python compute_features.py
cd ..
```

to obtain the image features (`local_entropy_[0-9]`, `cell_brightness_[0-9]`, `fractal_dimension_[0-9]`, `area_[0-9]`, `circumference_[0-9]`, `A_C_ratio_[0-9]`, `optical_flow_[0-8]`) for each image steam. These features will be saved as pandas DataFrame files (`./data/CD01-*/CD01-*_features.pkl'`); we also provide the `*.csv` file for an easy browse. 

Finally, we convert the features to a 21-D feature vector for each image stream, obtain their ground-truth labels from the experimental results, and get the final dataset (also saved as pandas DataFrames, `./data/dataset.pkl`). We further split the dataset into a training set (`./data/dataset_train.pkl`) and a test set (`./data/dataset_test.pkl`). Please refer to the jupyter notebook [./data/prepare_dataset.ipynb](./data/prepare_dataset.ipynb) in python for more detail.



**Note:**

Since the total size of the image stream data is extremely large (~52GB), we only provide the computed features for each image (`./data/CD01-*/CD01-*_features.pkl`). If you want to reproduce our feature extraction, please contact the corresponding author for the image stream data. 



## Feature visualization

We first visualize the feature space using Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) (**Fig. 4d, Supplementary Fig. S10c-e**). We provide the code for generating the LDA/PCA plots in the jupyter notebook [./feature_visualization.ipynb](./feature_visualization.ipynb).



## Machine learning

We next perform machine learning to classify each well as "low", "optimal" or "high" under CHIR duration = 24h, 36h, and 48h. We evaluate the classification performance by accuracy, precision, recall, F1-score, and AUC (**Fig. 4e, Supplementary Fig. S10f**). We also derive the feature importance weights by ANOVA (**Supplementary Fig. S10b**). To intervene in the differentiation condition in time, the classifier can also provide a "Deviation Score" for each CHIR dose (**Fig. 4f**). The predicted Deviation Scores can help select the optimal CHIR duration and improve the differentiation efficiency (**Fig. 4h**). The relevant codes are provided in the jupyter notebook [./machine_learning.ipynb](./machine_learning.ipynb). 

To test the model’s generalization ability to new batches, we conduct a cross-batch validation under a CHIR duration of 24h (**Fig. 4g, Supplementary Fig. S10g, h**). The code for cross-batch validation is provided in the jupyter notebook [./cross_batch_validation.ipynb](./cross_batch_validation.ipynb).

