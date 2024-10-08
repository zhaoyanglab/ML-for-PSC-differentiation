brightfield_path = '../data/Test';
result_path_patches = '../img_result';
result_path_merged = '../img_result_wholewell';
num_patches = [19 19];
patch_size = [512 512];
steps = [128 128];

nr = (num_patches(1) - 1) * steps(1) + patch_size(1);
nc = (num_patches(2) - 1) * steps(2) + patch_size(2);
result_heatmap = zeros(nr, nc, 3);
weight = zeros(nr, nc);
result_binary = zeros(nr, nc, 'uint8');

batch_dirs = dir(fullfile(brightfield_path, 'CD*'));
for i = 1:length(batch_dirs)
    batch_name = batch_dirs(i).name;
    mkdir(fullfile(result_path_merged, 'binary', batch_name));
    mkdir(fullfile(result_path_merged, 'heatmap', batch_name));
    
    img_dirs = dir(fullfile(brightfield_path, batch_name, 'Brightfield/*.png'));
    for j = 1:length(img_dirs)

        img_name = img_dirs(j).name;
        fprintf('Processing %s %s\n', batch_name, img_name);

        result_heatmap(:, :, :) = 0;
        weight(:, :) = 0;
        result_binary(:, :) = 0;
        
        for r_id = 1:num_patches(1)
            for c_id = 1:num_patches(2)
                patch_id = (r_id - 1) * num_patches(2) + c_id;

                patch_folder = dir(fullfile(result_path_patches, ...
                    sprintf('*_%s_%s_%d', batch_name, img_name, patch_id)));
                patch_folder = fullfile(patch_folder(1).folder, patch_folder(1).name);

                patch_heatmap = double(imread(fullfile(patch_folder, 'gradcam_cam.jpg')));
                patch_bw = imread(fullfile(patch_folder, 'gradcam_cam_binary.jpg'));

                r1 = (r_id - 1) * steps(1) + 1; r2 = r1 + patch_size(1) - 1;
                c1 = (c_id - 1) * steps(2) + 1; c2 = c1 + patch_size(2) - 1;
                
                weight(r1:r2, c1:c2) = weight(r1:r2, c1:c2) + 1;
                result_heatmap(r1:r2, c1:c2, :) = result_heatmap(r1:r2, c1:c2, :) + patch_heatmap;
                result_binary(r1:r2, c1:c2) = result_binary(r1:r2, c1:c2) + patch_bw;
            end
        end

        result_heatmap_img = cast(result_heatmap ./ weight, 'uint8');
        imwrite(result_heatmap_img, fullfile(result_path_merged, 'heatmap', batch_name, img_name));
        
        imwrite(result_binary, fullfile(result_path_merged, 'binary', batch_name, img_name));
    end
end