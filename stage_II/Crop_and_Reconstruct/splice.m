fullimg_path = 'E:\MyData\PHD\WeaklySupervisedLearning\dd\result-all_1';        % input path
subpath_fullimg = dir(fullimg_path);
fullimg_num = length(subpath_fullimg)-2;
for j = 1:length(fullimg_num)
    fullimg = subpath_fullimg(j+2).name;
    imgpath = fullfile(fullimg_path,fullimg);
    subpath = dir(imgpath);
    img_num = length(subpath) - 2;
    overlap = 128;
    spacing = 256;
    origin_img = zeros(overlap*(sqrt(img_num)-1)+spacing,overlap*(sqrt(img_num)-1)+spacing);
    layer_img = zeros(overlap*(sqrt(img_num)-1)+spacing,overlap*(sqrt(img_num)-1)+spacing);
    img_row = 1;
    img_col = 1;
    for i = 1:img_num
        id = sprintf('_%d',i);
        imgname = [fullimg,id,'_fake_B.png'];
        Imgpath = fullfile(subpath(i+2).folder,imgname);
        img = double(imread(Imgpath));
        origin_img_0 = zeros(overlap*(sqrt(img_num)-1)+spacing,overlap*(sqrt(img_num)-1)+spacing);
        layer_img_0 = zeros(overlap*(sqrt(img_num)-1)+spacing,overlap*(sqrt(img_num)-1)+spacing);
        origin_img_0(1+(img_row-1)*overlap:spacing+(img_row-1)*overlap,1+(img_col-1)*overlap:spacing+(img_col-1)*overlap) = img;
        layer_img_0(1+(img_row-1)*overlap:spacing+(img_row-1)*overlap,1+(img_col-1)*overlap:spacing+(img_col-1)*overlap) = ones(size(img,1),size(img,2));
        
        origin_img = origin_img + origin_img_0;
        layer_img = layer_img + layer_img_0;
        if img_col < sqrt(img_num)
            img_col = img_col + 1;
        elseif img_col == sqrt(img_num)
            img_col = 1;
            img_row = img_row +1;
        end
    end
    origin_img = origin_img./layer_img;
    % imgr = origin_img(:,:,1);
    % imgg = origin_img(:,:,2);
    % imgb = origin_img(:,:,3);
    % origin_imgr = imgaussfilt(imgr,5);
    % origin_imgg = imgaussfilt(imgg,5);
    % origin_imgb = imgaussfilt(imgb,5);
    % origin_img = cat(3,origin_imgr,origin_imgg,origin_imgb);
    
    
    % origin_img_gray = rgb2gray(uint8(origin_img));
    % imshow(origin_img_gray);
    % origin_img = im2bw(origin_img,10);
    
    % origin_img(257:size(origin_img,1)-256,257:size(origin_img,2)-256,:) = origin_img(257:size(origin_img,1)-256,257:size(origin_img,2)-256,:)/4;
    % origin_img(257:size(origin_img,1)-256,1:256,:) = origin_img(257:size(origin_img,1)-256,1:256,:)/2;
    % origin_img(1:256,257:size(origin_img,2)-256,:) = origin_img(1:256,257:size(origin_img,2)-256,:)/2;
    % origin_img(257:size(origin_img,1)-256,size(origin_img,2)-256:size(origin_img,2),:) = origin_img(257:size(origin_img,1)-256,size(origin_img,2)-256:size(origin_img,2),:)/2;
    % origin_img(size(origin_img,1)-256:size(origin_img,1),257:size(origin_img,2)-256,:) = origin_img(size(origin_img,1)-256:size(origin_img,1),257:size(origin_img,2)-256,:)/2;
    % imshow(origin_img);
    imwrite(uint8(origin_img),['E:\MyData\PHD\WeaklySupervisedLearning\dd\result_splice\',fullimg,'.png']);
end


