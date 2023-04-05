% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
function [pixels,pixel_metadata,label_metadata,other_metadata] = LoadCLEVRPos(spec, is_trn)
    arguments
        spec(1,1) string, is_trn(1,1) logical
    end
    assert(spec == "clevrpossimple");
    
    pixel_metadata = struct();
    label_metadata = struct();
    other_metadata = struct();

    if is_trn
        s = jsondecode(fileread(fullfile(Config.DATASET_DIR, "custom_clevr", "customclevr_trnsimple_config.json")));
    else
        s = jsondecode(fileread(fullfile(Config.DATASET_DIR, "custom_clevr", "customclevr_tstsimple_config.json")));
    end

    label_metadata.colors = fieldnames(s.colors);
    label_metadata.pixel_coords_r = s.pixel_coords_y'; % now n_objects x n_images
    label_metadata.pixel_coords_c = s.pixel_coords_x'; % now n_objects x n_images
    n_colors = numel(label_metadata.colors);
    bucketr = 0:10:240; % there are 25 of these
    bucketc = 0:10:320; % there are 33 of these
    [pixelr,pixelc] = meshgrid(bucketc, bucketr); % yes matlab is backwards
    pixel_metadata.chanidx = zeros(numel(bucketr), numel(bucketc), n_colors);
    pixel_metadata.bucketr = zeros(numel(bucketr), numel(bucketc), n_colors);
    pixel_metadata.bucketc = zeros(numel(bucketr), numel(bucketc), n_colors);
    nodestates             = zeros(numel(bucketr), numel(bucketc), n_colors, s.n_images, "logical");
    nodeobjidxstate        = zeros(numel(bucketr), numel(bucketc), n_colors, s.n_images);
    for clridx = 1 : n_colors
        pixel_metadata.chanidx(:,:,clridx) = clridx;
        pixel_metadata.bucketr(:,:,clridx) = pixelr;
        pixel_metadata.bucketc(:,:,clridx) = pixelc;
        for imgidx = 1 : s.n_images
            for objidx = 1 : s.n_objects
                if strcmp(s.color_name{imgidx}{objidx}, label_metadata.colors{clridx})
                    r = floor(label_metadata.pixel_coords_r(objidx,imgidx)/10);
                    c = floor(label_metadata.pixel_coords_c(objidx,imgidx)/10);
                    nodestates(r,c,clridx,imgidx) = true;
                    nodeobjidxstate(r,c,clridx,imgidx) = objidx;
                end
            end
        end
    end
    % pixels = for each color channel clr1, for each color channel clr2, is there an obj in relative location bucket [x,y]?
    pixels = reshape(nodestates, [], s.n_images);
    pixel_metadata.chanidx = reshape(pixel_metadata.chanidx, [], 1); % color channel
    pixel_metadata.bucketr = reshape(pixel_metadata.bucketr, [], 1);
    pixel_metadata.bucketc = reshape(pixel_metadata.bucketc, [], 1);
    pixel_metadata.name = cell(size(pixels, 1), 1);
    for i = 1 : size(pixels, 1) % for each pixel
        pixel_metadata.name{i} = ['r',num2str(pixel_metadata.bucketr(i)),'c',num2str(pixel_metadata.bucketc(i)),'clr',num2str(pixel_metadata.chanidx(i))];
    end
    label_metadata.nodeobjidxstate = reshape(nodeobjidxstate, [], s.n_images); % n_pixels x n_images - object at this pixel (if any)
    label_metadata.is_face = (mod((1:s.n_images)', 2) == 1);  % n_images x 1
    label_metadata.eyes_same_color = s.eyes_same_color;       % n_images x 1
    label_metadata.randomized_obj_idx = s.randomized_obj_idx; % n_images x 1
    other_metadata.n_objects = s.n_objects;
end

