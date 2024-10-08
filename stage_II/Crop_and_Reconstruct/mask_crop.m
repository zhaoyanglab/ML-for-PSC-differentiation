ImageCrop('../data/Train', '../data/Train_crop', 'png', 512, 256);
ImageCrop('../data/Test', '../data/Test_crop', 'png', 512, 128);

function ImageCrop(src_dir,dst_dir,format,spacing,step)

    batch_dirs = dir(fullfile(src_dir, 'CD*'));
    img_types = {'Brightfield', 'Mask'};
    mkdir(fullfile(dst_dir, 'Brightfield'));
    mkdir(fullfile(dst_dir, 'Mask'));
    
    for ii = 1:length(batch_dirs)
        batchname = batch_dirs(ii).name;
        fprintf('Processing Batch %s (%s)\n', batchname, src_dir);
    
        for jj = 1:numel(img_types)
            img_type = img_types{jj};
            imgdirs = dir(fullfile(src_dir, batchname, img_type, ['*.',format]));

            for kk = 1:length(imgdirs)
                imgname = imgdirs(kk).name;
                fprintf('--> %s/%s\n', imgdirs(kk).folder, imgname);
                
                I = imread(fullfile(imgdirs(kk).folder, imgname));
                I = imresize(I,[2816,2816]);
                row = floor((size(I,1)-spacing)/step) + 1;
                col = floor((size(I,1)-spacing)/step) + 1;
                % Start Croping
                for rr = 1:row
                    for cc = 1:col
                        rect = [(cc-1)*step+1,(rr-1)*step+1,spacing-1,spacing-1];
                        newI = imcrop(I,rect);
                        if strcmp(img_type, 'Mask')
                            newI = newI * 60;
                            newI = uint8(newI);
                        end
                        newname = [batchname '_' imgname '_' num2str((rr-1)*col+cc) '.' format];
                        newpath = fullfile(dst_dir, img_type, newname);
                        imwrite(newI,newpath);
                    end
                end
            end
        end
    end
end
