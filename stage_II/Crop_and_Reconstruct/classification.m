mainpath = 'Z:\User\zhaoyang\g\test_img_crop\mask\mask_SC002';
imgpath = dir(mainpath);
for i = 1:length(imgpath)-2
    Imgpath = fullfile(imgpath(i+2).folder,imgpath(i+2).name);
    img = imread(Imgpath);
    pos = find(img == 60);
    neg = find(img == 0);
    if length(pos) > 0.3*numel(img)
        imgpath_new = ['Z:\User\zhaoyang\g\test_img_crop_1\mask\SC002_1\',imgpath(i+2).name];
        imwrite(img,imgpath_new);
    elseif length(neg) == numel(img)
        imgpath_new = ['Z:\User\zhaoyang\g\test_img_crop_0\mask\SC002_0\',imgpath(i+2).name];
        imwrite(img,imgpath_new);
    end
end