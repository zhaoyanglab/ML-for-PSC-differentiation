data_dir = '../data';
splits = {'Train', 'Test'};

for t = 1:2
    split = splits{t};

    ret = {'filepath' 'target'};

    brightfield_imgs = dir(fullfile(data_dir, [split '_crop'], 'Brightfield', '*.png'));
    brightfield_imgs = sortfiles(brightfield_imgs);
    mask_imgs = dir(fullfile(data_dir, [split '_crop'], 'Mask', '*.png'));
    mask_imgs = sortfiles(mask_imgs);

    for i = 1:length(brightfield_imgs)
        mask_img = imread(fullfile(mask_imgs(i).folder, mask_imgs(i).name));

        pos_pixels = sum(sum(mask_img == 60));
        neg_pixels = sum(sum(mask_img == 0));
        num_pixels = numel(mask_img);

        if (pos_pixels > 0.3 * num_pixels)
            ret = [ret; {fullfile(brightfield_imgs(i).folder, brightfield_imgs(i).name) 1}];
        elseif (neg_pixels == num_pixels)
            ret = [ret; {fullfile(brightfield_imgs(i).folder, brightfield_imgs(i).name) 0}];
        elseif strcmp(split, 'Test')
            ret = [ret; {fullfile(brightfield_imgs(i).folder, brightfield_imgs(i).name) 2}];
        end
    end

    writecell(ret, fullfile(data_dir, [split '.csv']), 'Delimiter', ',');
end

function ret = sortfiles(files)
    ret = table2struct(sortrows(struct2table(files), 'name'));
end