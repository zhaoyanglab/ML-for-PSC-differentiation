mainpath_ctnt = 'E:\MyData\PHD\WeaklySupervisedLearning\新数据\aug\aug\CD13';        % Path of the cTNT fluorescent image
imgpath_ctnt = dir(mainpath_ctnt);
for i = 1:length(imgpath_ctnt) - 2
    Imgpath_ctnt = fullfile(imgpath_ctnt(i+2).folder,imgpath_ctnt(i+2).name);
    differentiation_rate(i) = differentiation(Imgpath_ctnt);
end

mainpath_predict = 'E:\MyData\PHD\WeaklySupervisedLearning\新数据\CD13\img_BW_splice\NotBad';    % Path of the predicted binary image
imgpath_predict = dir(mainpath_predict);
for i = 1:length(imgpath_predict) - 2
    Imgpath_predict = fullfile(imgpath_predict(i+2).folder,imgpath_predict(i+2).name);
    img_predict = double(imread(Imgpath_predict));
    differentiation_rate_pre(i) = length(find(img_predict == 255))/numel(img_predict);
%     differentiation_rate_pre(i) = differentiation(Imgpath_predict);
end
% Pearson correlation
[R,P] = corrcoef(differentiation_rate',differentiation_rate_pre');
% figure
p = polyfit(differentiation_rate', differentiation_rate_pre', 1); % fitted coeffcient p
yFit = polyval(p, differentiation_rate'); % estimate fitted Y with X and p.
figure(9),
set(gcf,'InvertHardCopy','off','color','white');
hold all,
scatter(differentiation_rate', differentiation_rate_pre');
plot(differentiation_rate',yFit,'-');
xlabel('differentiation rate');
ylabel('differentiation rate pre');
% title(sprintf('p = %g', pval));

function differentiation_rate = differentiation(img_dir)
img = im2double(imread(img_dir));
% img = img(1:5632,1:5632);
img = imresize(img,[2816,2816]);
% img_norm = mat2gray(img);
img(img > 0.6) = 0;
level = 0.15;
differentiation_rate = sum(img(img>level))/numel(img);

% img_bw = im2bw(img_norm,level);
% img_bw = img_bw(1:5632,1:5632);
% figure(1);
% imshow(img_bw);
% figure(2);
% imshow(img_norm);
% differentiation_rate = length(find(img_bw == 1))/numel(img_bw);
end