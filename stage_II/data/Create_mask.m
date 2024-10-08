% Create a UI window for image selection
[selected_file, file_path] = uigetfile({'*.jpg;*.png;*.tif', 'Image Files (*.jpg, *.png, *.tif)'}, 'Select an Image');
imgpath = fullfile(file_path, selected_file);
imgsize = size(imread(imgpath));
imds = imageDatastore(imgpath);
imageLabeler(imds);

% Wait for the user to finish labeling
fprintf("Press any key to continue...");
pause;

% Get the labeled ROI data
out = gatherLabelData(gTruth,[labelType.Polygon],'GroupLabelData','LabelName');

positive_polygons = out{1}.positive{1};
if isa(positive_polygons, 'double')
    positive_polygons = {positive_polygons};
end

possible_polygons = out{1}.possible{1};
if isa(possible_polygons, 'double')
    possible_polygons = {possible_polygons};
end

negative_polygons = out{1}.negative{1};
if isa(negative_polygons, 'double')
    negative_polygons = {negative_polygons};
end

% Create the mask image for possible ROIs and positive ROIs.
mask_positive = poly2label(positive_polygons', ones(1, length(positive_polygons)), imgsize);
mask_possible = poly2label(possible_polygons', ones(1, length(possible_polygons)), imgsize);
mask_negative = poly2label(negative_polygons', ones(1, length(negative_polygons)), imgsize);

mask = mask_possible * 2;
mask(mask_positive > 0) = 1;
mask(mask_negative > 0) = 0;
mask = cast(mask, 'uint8');

% Show the mask 
imshow(mask * 60); hold on;
patch([0 0 0], [0 0 0], [60 60 60]/255, 'EdgeColor', 'none');
patch([0 0 0], [0 0 0], [120 120 120]/255, 'EdgeColor', 'none');
patch([0 0 0], [0 0 0], [0 0 0], 'EdgeColor', 'none');
legend('positive', 'possible', 'negative', 'Location', 'NorthEastOutside');

% Save the mask image
parts = strsplit(selected_file, '~');
if length(parts) >= 2
    savename = strcat(parts{1}, '_label.png');
else
    savename = 'label.png';
end
mkdir(fullfile(file_path, '../Mask'));
[filename, pathname] = uiputfile({'*.png', 'PNG Files (*.png)'}, 'Save Mask As', fullfile(file_path, '../Mask', savename));
if isequal(filename,0)
    disp('Please select the path for saving the mask. You may also type `imwrite(mask, xxxxxx)` (xxxxxx is the filename) in the MATLAB command line window to save the mask image.');
    return;
end
imwrite(mask, fullfile(pathname, filename));
disp('Mask image saved successfully.');
