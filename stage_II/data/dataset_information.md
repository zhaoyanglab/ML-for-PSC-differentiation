# Dataset Information

The dataset contains 141 pairs of whole-well bright-field images at day 6 and manually labeled masks (`./{Train|Test}/CD02-*/{Brightfield|Mask}/*.png`), where the training set and the test set contain 106 and 35 pairs, respectively. The corresponding cTnT fluorescence results for the test images are also provided (`./Test/CD02-*/cTnT/*.png`).  

The mask images contains pixels with intensity 0 (negative), 1 (positive), or 2 (possible). We used Image Labeler in MATLAB to create these masks. An example script is provided in `Create_mask.m`. By executing the script, you should follow the steps below:

1. Select a bright-field image to be annotated;
2. In the Image Labeler GUI window, manually define labels (0->negative, 1->positive, 2->possible);
3. Annotate negative, positive, and possible regions using polygon tools;
4. Export these annotated regions to the workspace, and close the Image Labeler GUI window;
5. Select the folder to save the mask image.

Repeatedly run the script to label all the bright-field images. During labeling the images, you should refer to the final cTnT fluorescence results and track the cTnT+ cells from day 12 back to day 6.