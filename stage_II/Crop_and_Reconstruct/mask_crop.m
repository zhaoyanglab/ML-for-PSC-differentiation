ImageCrop('E:\MyData\PHD\WeaklySupervisedLearning\dd\bigimage','E:\MyData\PHD\WeaklySupervisedLearning\dd\bigimage_crop','png',512,256);

function ImageCrop(src_dir,dst_dir,format,spacing,overlap)

subfolders = dir(src_dir);
for ii = 1:length(subfolders)
    subname = subfolders(ii).name;
    % ignore current dir and father dir
    if ~strcmp(subname,'.')&&~strcmp(subname,'..')
        frames = dir(fullfile(src_dir,subname,['*.',format]));
        imgnum = length(frames);
        dstpath = fullfile(dst_dir,subname);
        if ~isdir(dstpath)
            mkdir(dstpath);
        end
        for jj=1:imgnum
            imgpath = fullfile(src_dir,subname,frames(jj).name);
            I = imread(imgpath);
            I = imresize(I,[2816,2816]);
%             I = I*60;
            row = floor((size(I,1)-spacing)/overlap) + 1;
            col = floor((size(I,1)-spacing)/overlap) + 1;
            % Start Croping
            for rr = 1:row
                for cc = 1:col
                    rect = [(cc-1)*overlap+1,(rr-1)*overlap+1,spacing-1,spacing-1];
                    newI = imcrop(I,rect);
                    newI = uint8(newI);
                    newname = [frames(jj).name,'_',num2str((rr-1)*col+cc),['.',format]];
                    newpath = fullfile(dstpath,newname);
                    imwrite(newI,newpath);
                end
            end
        end
    end % end of if
end
end
