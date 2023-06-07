% Code for comparing the true and the predicted fluorescence images at the pixel level 

img_size = 512;
img_gt = imread('./typical_examples/true/CD03-2-1.png'); % The true fluorescence image
img_pre = imread('./typical_examples/predicted/CD03-2-1.png'); % The predicted fluorescence image

img_gt = imresize(im2double(img_gt),[img_size,img_size]);
img_pre = imresize(im2double(img_pre),[img_size,img_size]);

norm_gt = img_gt;
norm_pre = img_pre;

error_map = norm_gt - norm_pre;
error_map_pos = error_map;
error_map_pos(error_map_pos<0) = 0;
error_map_pos_rgb = zeros(img_size,img_size,3);
error_map_pos_rgb(:,:,1) = error_map_pos;
error_map_neg = error_map;
error_map_neg(error_map_neg>0) = 0;
error_map_neg_rgb = zeros(img_size,img_size,3);
error_map_neg_rgb(:,:,3) = abs(error_map_neg);
error_map_rgb = error_map_neg_rgb + error_map_pos_rgb;

% figure(1);
% imshow(norm_gt);
% figure(2);
% imshow(norm_pre);
% figure(3);
% imshow(error_map_rgb);
% figure(5);
% a = histogram(norm_gt);
% figure(6);
% b = histogram(norm_pre);
col_gt = reshape(norm_gt,1,img_size*img_size);
col_pre = reshape(norm_pre,1,img_size*img_size);

norm_gt_ds = zeros(img_size,img_size);
norm_gt_ds(norm_gt <= 0.1) = 0.05;
norm_gt_ds(norm_gt > 0.1 & norm_gt <= 0.2) = 0.15;
norm_gt_ds(norm_gt > 0.2 & norm_gt <= 0.3) = 0.25;
norm_gt_ds(norm_gt > 0.3 & norm_gt <= 0.4) = 0.35;
norm_gt_ds(norm_gt > 0.4 &  norm_gt <= 0.5) = 0.45;
norm_gt_ds(norm_gt > 0.5 & norm_gt <= 0.6) = 0.55;
norm_gt_ds(norm_gt > 0.6 & norm_gt <= 0.7) = 0.65;
norm_gt_ds(norm_gt > 0.7 & norm_gt <= 0.8) = 0.75;
norm_gt_ds(norm_gt > 0.8 & norm_gt <= 0.9) = 0.85;
norm_gt_ds(norm_gt > 0.9) = 0.95;
col_gt_ds = reshape(norm_gt_ds,1,img_size*img_size);

norm_pre_ds = zeros(img_size,img_size);
norm_pre_ds(norm_pre <= 0.1) = 0.05;
norm_pre_ds(norm_pre > 0.1 & norm_pre <= 0.2) = 0.15;
norm_pre_ds(norm_pre > 0.2 & norm_pre <= 0.3) = 0.25;
norm_pre_ds(norm_pre > 0.3 & norm_pre <= 0.4) = 0.35;
norm_pre_ds(norm_pre > 0.4 & norm_pre <= 0.5) = 0.45;
norm_pre_ds(norm_pre > 0.5 & norm_pre <= 0.6) = 0.55;
norm_pre_ds(norm_pre > 0.6 & norm_pre <= 0.7) = 0.65;
norm_pre_ds(norm_pre > 0.7 & norm_pre <= 0.8) = 0.75;
norm_pre_ds(norm_pre > 0.8 & norm_pre <= 0.9) = 0.85;
norm_pre_ds(norm_pre > 0.9) = 0.95;
col_pre_ds = reshape(norm_pre_ds,1,img_size*img_size);

corr_matrix = zeros(10,10);
for i = 1:img_size*img_size
    corr_matrix(11-uint8((col_pre_ds(i)+0.05)*10),uint8((col_gt_ds(i)+0.05)*10)) = corr_matrix(11-uint8((col_pre_ds(i)+0.05)*10),uint8((col_gt_ds(i)+0.05)*10)) + 1;
end
corr_matrix = corr_matrix/100;

figure(4);
image(corr_matrix);

xticks([1 2 3 4 5 6 7 8 9 10])
xticklabels({'0.05','0.15','0.25','0.35','0.45','0.55','0.65','0.75','0.85','0.95'})
yticks([1 2 3 4 5 6 7 8 9 10])
yticklabels({'0.95','0.85','0.75','0.65','0.55','0.45','0.35','0.25','0.15','0.05'})
set(gca, 'FontName', 'Arial', 'FontSize', 18, ...
    'XTickLabelRotation', 0)

xlabel('true intensity','FontSize',25,'FontName','Arial');
ylabel('predicted intensity','FontSize',25,'FontName','Arial');

set(gcf,'Units','centimeter','Position',[10 10 18 16]);

for i = 1:10
    for j = 1:10
        num = int2str(corr_matrix(j,i));
        if round(corr_matrix(j,i)) >= 20
            text(i,j,num,'FontName','Arial','FontSize',20,'FontWeight','bold','HorizontalAlignment','center','Color','k');
        else
            text(i,j,num,'FontName','Arial','FontSize',20,'FontWeight','bold','HorizontalAlignment','center','Color','w');
        end
    end
end
hold on;
for i = 1:10
   plot([.5,10.5],[i-.5,i-.5],'w-','linewidth',1);
   plot([i-.5,i-.5],[.5,10.5],'w-','linewidth',1);
end
% set(gca,'colorscale','log');
colorbar('off');

[R,P] = corrcoef(col_gt,col_pre);
fprintf("Pixel Correlation = %.6f\n", R(1, 2));



