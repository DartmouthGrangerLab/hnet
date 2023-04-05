% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
function [pixels,pixel_metadata,label_metadata,other_metadata] = LoadCLEVR(spec, is_trn)
    arguments
        spec(1,1) string, is_trn(1,1) logical
    end
    assert(spec == "clevr");

    pixel_metadata = struct();
    label_metadata = struct();
    other_metadata = struct();

    other_metadata = CachedCompute(@LoadAndBinarizeCLEVR, is_trn);

    % for performance, drop all but 100 images
    drop = true(other_metadata.n, 1);
    drop(1:100) = false;
    other_metadata.image_idx(drop) = [];
    other_metadata.objects(drop)   = [];
    other_metadata.img(:,:,:,drop) = [];
    other_metadata.n = sum(~drop);

    label_metadata.image_idx = other_metadata.image_idx;
    other_metadata = rmfield(other_metadata, "image_idx");
    
    pixels = reshape(other_metadata.img, [], size(other_metadata.img, 4));

    pixel_metadata.name  = cell(size(pixels, 1), 1);
    for i = 1 : size(pixels, 1)
        pixel_metadata.name{i} = ''; % for now
    end
    chanidx = zeros(size(other_metadata.img, 1), size(other_metadata.img, 2), size(other_metadata.img, 3));
    for i = 1 : size(other_metadata.img, 3)
        chanidx(:,:,i) = i;
    end
    pixel_metadata.chanidx = reshape(chanidx, [], size(other_metadata.img, 4));
end
