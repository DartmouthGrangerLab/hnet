% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% object colors are: red, green, blue, yellow, purple, cyan, gray, brown
function dat = LoadAndBinarizeCLEVR(is_trn)
    if is_trn
        dat = io.LoadCLEVR('train', 0.25);
    else
        dat = io.LoadCLEVR('val', 0.25);
    end
    
    % remove unneeded fields to save memory and reduce load time
    dat = rmfield(dat, 'questions');
    dat = rmfield(dat, 'relationships_right');
    dat = rmfield(dat, 'relationships_behind');
    dat = rmfield(dat, 'relationships_front');
    dat = rmfield(dat, 'relationships_left');
    dat = rmfield(dat, 'directions_right');
    dat = rmfield(dat, 'directions_behind');
    dat = rmfield(dat, 'directions_above');
    dat = rmfield(dat, 'directions_below');
    dat = rmfield(dat, 'directions_left');
    dat = rmfield(dat, 'directions_front');
    dat = rmfield(dat, 'image_filename');
    
    % remove images with gray or brown objects
    keep = true(dat.n, 1); % keeps 13271 out of 70K
    for i = 1 : dat.n
        for j = 1 : numel(dat.objects{i})
            if strcmp(dat.objects{i}(j).color, 'gray') || strcmp(dat.objects{i}(j).color, 'brown')
                keep(i) = false;
            end
        end
    end
    dat.n = sum(keep);
    dat.image_idx(~keep) = [];
    dat.objects(~keep)   = [];
    dat.img(:,:,:,~keep) = [];

    img = im2double(dat.img);
    dat = rmfield(dat, 'img');
    
    dat.chan_names = {'red','green','blue','yellow','magenta','cyan'};
    way1img = zeros(size(img, 1), size(img, 2), numel(dat.chan_names), dat.n);
    way1img(:,:,1,:) = max(0, (img(:,:,1,:) - (img(:,:,2,:) + img(:,:,3,:))./2)); % red vs rest images
    way1img(:,:,2,:) = max(0, (img(:,:,2,:) - (img(:,:,1,:) + img(:,:,3,:))./2));
    way1img(:,:,3,:) = max(0, (img(:,:,3,:) - (img(:,:,1,:) + img(:,:,2,:))./2));
    way1img(:,:,4,:) = max(0, (img(:,:,1,:) + img(:,:,2,:))./2 - img(:,:,2,:));
    way1img(:,:,5,:) = max(0, (img(:,:,1,:) + img(:,:,3,:))./2 - img(:,:,2,:)); % hopefully covers "purple"
    way1img(:,:,6,:) = max(0, (img(:,:,2,:) + img(:,:,3,:))./2 - img(:,:,1,:));
    for i = 1 : numel(dat.chan_names)
        way1img(:,:,i,:) = way1img(:,:,i,:) ./ max(way1img(:,:,i,:), [], 'all'); % first, normalize each color channel
    end
    for i = 1 : dat.n
        way1img(:,:,:,i) = way1img(:,:,:,i) ./ max(way1img(:,:,:,i), [], 'all'); % second, normalize each image competitively
    end
    for i = 1 : 20
        imwrite(img(:,:,:,i), ['./clevr_img',num2str(i),'.png']);
    end
    clearvars img
    
    temp = zeros(size(way1img, 1), size(way1img, 2), numel(dat.chan_names), dat.n);
    
    thresh = 0.45;
    for i = 1 : dat.n
%     [~,idx] = min(way1img(:,:,:,i), [], 3); % min because this is distance
        [~,idx] = max(way1img(:,:,:,i), [], 3); % max because this is similarity
        code = encode.OneHot(idx(:), size(way1img, 3));
        for j = 1 : numel(dat.chan_names)
            temp(:,:,j,i) = reshape(code(:,j), size(idx, 1), size(idx, 2));
        end
    end
%     temp(way1img > thresh) = 0; % if this is distance
    temp(way1img < thresh) = 0; % if this is similarity
    clearvars way1img

    left   = zeros(size(temp, 1), size(temp, 2), numel(dat.chan_names), dat.n, 'logical');
    right  = zeros(size(temp, 1), size(temp, 2), numel(dat.chan_names), dat.n, 'logical');
    top    = zeros(size(temp, 1), size(temp, 2), numel(dat.chan_names), dat.n, 'logical');
    bottom = zeros(size(temp, 1), size(temp, 2), numel(dat.chan_names), dat.n, 'logical');

    left(:,1:end-1,:,:)   = temp(:,1:end-1,:,:) < temp(:,2:end,:,:);
    right(:,1:end-1,:,:)  = temp(:,2:end,:,:) < temp(:,1:end-1,:,:);
    top(1:end-1,:,:,:)    = temp(1:end-1,:,:,:) < temp(2:end,:,:,:);
    bottom(1:end-1,:,:,:) = temp(2:end,:,:,:) < temp(1:end-1,:,:,:);
    clearvars temp
    % verified with:
%     i=4;imshow(cat(3, left(:,:,3,i), right(:,:,3,i), right(:,:,3,i)))
%     i=4;imshow(cat(3, top(:,:,3,i), bottom(:,:,3,i), bottom(:,:,3,i)))

    dat.img = cat(3, left, right, top, bottom);
    dat.chan_color = repmat(dat.chan_names, 1, 4);
    dat.chan_dir = cat(2, repmat({'left'}, 1, numel(dat.chan_names)), repmat({'right'}, 1, numel(dat.chan_names)), repmat({'top'}, 1, numel(dat.chan_names)), repmat({'bottom'}, 1, numel(dat.chan_names)));
end