img_path = 'E:\MyData\PHD\WeaklySupervisedLearning\新数据\CD58A\S11';           % input path 
subpath = dir(img_path);
img_num = length(subpath)-2;
overlap = 128;
spacing = 512;
origin_img = zeros(overlap*(sqrt(img_num)-1)+spacing,overlap*(sqrt(img_num)-1)+spacing);
img_row = 1;
img_col = 1;
for i = 1:img_num
    Imgname = sprintf('S11~Z3.png_%d',i);
    imgname = 'gradcam_grayscale_cam.jpg';
    Imgpath = fullfile(subpath(i+2).folder,Imgname,imgname);
    img = double(imread(Imgpath));
    origin_img_0 = zeros(overlap*(sqrt(img_num)-1)+spacing,overlap*(sqrt(img_num)-1)+spacing);
    origin_img_0(1+(img_row-1)*overlap:spacing+(img_row-1)*overlap,1+(img_col-1)*overlap:spacing+(img_col-1)*overlap) = img;
    origin_img = origin_img + origin_img_0;
    if img_col < sqrt(img_num)
        img_col = img_col + 1;
    elseif img_col == sqrt(img_num)
        img_col = 1;
        img_row = img_row +1;
    end
end
% imshow(origin_img);
imwrite(uint8(origin_img),'E:\MyData\PHD\WeaklySupervisedLearning\新数据\CD58A\S11\S11~Z3_cam_BW1.png');


