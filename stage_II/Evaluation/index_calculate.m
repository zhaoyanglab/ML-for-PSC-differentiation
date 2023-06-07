img_pre_path = dir('E:\MyData\PHD\WeaklySupervisedLearning\dd\result\pre_1');       % Path of the predicted binarized image
img_path = dir('E:\MyData\PHD\WeaklySupervisedLearning\dd\result\gt_1');            % Path of the manually labeling mask
img_num = length(img_path) - 2;
for i = 1:img_num
    img_pre = fullfile(img_pre_path(i+2).folder,img_pre_path(i+2).name);
    img = fullfile(img_path(i+2).folder,img_path(i+2).name);
    D5_mask = im2double(imread(img));
    D5_mask = imresize(D5_mask,[2816,2816]);
    D5_mask_pre = im2double(imread(img_pre));
    D5_mask_pre = imresize(D5_mask_pre,[2816,2816]);  
    D5 = D5_mask + D5_mask_pre;
    Positive = length(find(D5_mask == 1));
    Negative = length(find(D5_mask == 0));
    Pre_Positive = length(find(D5_mask_pre == 1));
    Pre_Negative = length(find(D5_mask_pre == 0));
    Intersection = length(find(D5 == 2));
    Intersection_N = length(find(D5 == 0));
    Union = Intersection +  length(find(D5 == 1));
    IOU(i) = Intersection/Union;
    Ture_Positive(i) = Intersection/Positive;
    False_Positive(i) = (Pre_Positive - Intersection)/Negative;
    ACC(i) = (Intersection + Intersection_N)/(Positive + Negative);
    PPV(i) = Intersection/Pre_Positive;
    TNR(i) = Intersection_N/Negative;
    F1(i) = 2/(1/Ture_Positive(i) + 1/PPV(i));
end
