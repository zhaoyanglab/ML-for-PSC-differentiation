result = {'Batch_ID' 'S_id' 'predicted proportion of CM-committed CPCs' 'true differentiation efficiency'};
batch_paths = dir('../img_result_wholewell/binary/CD*');

fluorescence_thres = 0.5;
n = 0;
for batch_id = 1:numel(batch_paths)
    batch_name = batch_paths(batch_id).name;

    mask_predicted_paths = dir(fullfile('../img_result_wholewell/binary', batch_name, '*.png'));
    mask_predicted_paths = sortfiles(mask_predicted_paths);

    ctnt_paths = dir(fullfile('../data/Test', batch_name, 'cTnT/*.png'));
    ctnt_paths = sortfiles(ctnt_paths);

    for i = 1:numel(mask_predicted_paths)

        n = n + 1;

        img_name = mask_predicted_paths(i).name;
        % Path of the predicted binarized segmentation mask of CPC
        mask_pred_path = fullfile(mask_predicted_paths(i).folder, img_name);
        % Path of the manually labeled mask
        ctnt_path = fullfile(ctnt_paths(i).folder,ctnt_paths(i).name);

        % The predicted mask
        mask_pred = imread(mask_pred_path);
        % Compute the proportion of CM-committed CPCs
        proportion_pred(n) = mean(mean(mask_pred == 255));

        % The cTnT fluorescence label
        ctnt = double(imread(ctnt_path)) / 255;
        ctnt = imresize(ctnt, [2816, 2816]);
        differentiation_efficiency(n) = sum(ctnt(ctnt > fluorescence_thres))/numel(ctnt);
    
        result = [result; { ...
                batch_name img_name proportion_pred(n) differentiation_efficiency(n) ...
        }];
    end

end

writecell(result, 'whole_well_comparion.csv', 'Delimiter', ',');

% Pearson correlation
[R,P] = corrcoef(differentiation_efficiency', proportion_pred');
fprintf('r = %.2f, P = %.6f\n', R(1, 2), P(1, 2));

% figure
p = polyfit(differentiation_efficiency', proportion_pred', 1); % fitted coeffcient p
yFit = polyval(p, differentiation_efficiency'); % estimate fitted Y with X and p.
figure;
set(gcf,'InvertHardCopy','off','color','white');
scatter(differentiation_efficiency', proportion_pred' * 100);
hold on;
plot(differentiation_efficiency',yFit * 100,'-');
xlabel('Differentiation efficiency index');
ylabel('Predicted % of CPC regions');
ylim([0 100]);


function ret = sortfiles(files)
    ret = table2struct(sortrows(struct2table(files), 'name'));
end
