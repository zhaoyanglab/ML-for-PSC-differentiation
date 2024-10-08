result = {'Batch_ID' 'S_id' 'accuracy' 'F1 score' 'precision' 'recall' 'specificity' 'IoU'};

n = 0;

eps = 0.01;
batch_paths = dir('../img_result_wholewell/binary/CD*');
for batch_id = 1:numel(batch_paths)
    batch_name = batch_paths(batch_id).name;
    
    mask_predicted_paths = dir(fullfile('../img_result_wholewell/binary', batch_name, '*.png'));
    mask_predicted_paths = sortfiles(mask_predicted_paths);
    
    mask_gt_paths = dir(fullfile('../data/Test', batch_name, 'Mask/*.png'));
    mask_gt_paths = sortfiles(mask_gt_paths);

    mkdir(fullfile('../img_result_wholewell/binary_gt', batch_name));
    
    for i = 1:numel(mask_predicted_paths)
        img_name = mask_predicted_paths(i).name;

        % Path of the predicted binarized segmentation mask of CPC
        mask_pred_path = fullfile(mask_predicted_paths(i).folder, img_name);
        % Path of the manually labeled mask
        mask_gt_path = fullfile(mask_gt_paths(i).folder,mask_gt_paths(i).name);

        % the manually labeled mask
        mask_gt = imread(mask_gt_path);
        mask_gt = imresize(mask_gt, [2816,2816]);
        mask_gt = double(mask_gt > 0); 
        imwrite(cast(mask_gt * 255, 'uint8'), fullfile('../img_result_wholewell/binary_gt', batch_name, mask_gt_paths(i).name));
        
        % the predicted mask
        mask_pred = double(imread(mask_pred_path) > 0);

        Positive_GT = sum(sum(mask_gt == 1));
        Negative_GT = sum(sum(mask_gt == 0));
        Positive_Pred = sum(sum(mask_pred == 1));
        Negative_Pred = sum(sum(mask_pred == 0));
        Positive_Intersection = sum(sum(mask_pred + mask_gt == 2));
        Negative_Intersection = sum(sum(mask_pred + mask_gt == 0));
        Union = Positive_Intersection + sum(sum(mask_pred + mask_gt == 1));

        % for pixel-wise measurements, a well in taken into account if it contains target cells
        if Positive_GT / numel(mask_gt) >= eps
            n = n + 1;
            IOU(n) = Positive_Intersection / Union; % intersection-over-union
            True_Positive(n) = Positive_Intersection/Positive_GT; % true positive rate (i.e., recall)
            ACC(n) = (Positive_Intersection + Negative_Intersection)/(Positive_GT + Negative_GT); % accuracy
            PPV(n) = Positive_Intersection/Positive_Pred; % positive predictive value (i.e., precision)
            TNR(n) = Negative_Intersection / Negative_GT; % true negative rate (i.e., specificity) 
            F1(n) = 2/(1/True_Positive(n) + 1/PPV(n)); % F1 score

            result = [result; { ...
                batch_name img_name ACC(n) F1(n) PPV(n) True_Positive(n) TNR(n) IOU(n) ...
                }];
        end
    end
end

writecell(result, 'pixel_comparison.csv', 'Delimiter', ',');
fprintf("Accuracy (%%) = %.1f ± %.1f\n", mean(ACC, 'omitNaN') * 100, std(ACC, 'omitNaN') * 100);
fprintf("F1 (%%) = %.1f ± %.1f\n", mean(F1, 'omitNaN') * 100, std(F1, 'omitNaN') * 100);
fprintf("Precision (%%) = %.1f ± %.1f\n", mean(PPV, 'omitNaN') * 100, std(PPV, 'omitNaN') * 100);
fprintf("Recall (%%) = %.1f ± %.1f\n", mean(True_Positive, 'omitNaN') * 100, std(True_Positive, 'omitNaN') * 100);
fprintf("Specificity (%%) = %.1f ± %.1f\n", mean(TNR, 'omitNaN') * 100, std(TNR, 'omitNaN') * 100);
fprintf("IoU (%%) = %.1f ± %.1f\n", mean(IOU, 'omitNaN') * 100, std(IOU, 'omitNaN') * 100);

function ret = sortfiles(files)
    ret = table2struct(sortrows(struct2table(files), 'name'));
end
