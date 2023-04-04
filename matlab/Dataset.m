% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
classdef Dataset
    properties (SetAccess=immutable)
        frontend_spec (1,:) char
        img_sz        (:,1) = [] % [n_rows,n_cols,n_chan] or empty if not an image dataset
        uniq_classes  (1,:) cell
    end
    properties
        pixels    (:,:)      % n_nodes x n 
        label_idx (:,1)      % n x 1
        node_name (:,1) cell % n_nodes x 1
        meta      (1,1) struct = struct()
        labeldata (1,1) struct
    end
    properties (Dependent)
        n_pts     % scalar (int-valued numeric) number of datapoints / images
        n_classes % scalar (int-valued numeric)
        n_nodes   % scalar (int-valued numeric) aka number of features, aka number of pixels (n_rows*n_cols*n_chan)
    end


    methods
        function obj = Dataset(frontendSpec, trnOrTst) % constructor
            arguments
                frontendSpec, trnOrTst
            end
            obj.frontend_spec = frontendSpec;
            spec = frontendSpec;
            if contains(frontendSpec, ".")
                temp = strsplit(frontendSpec, ".");
                spec = temp{1};
                n_per_class = Str2Double(temp{2}); % only used if trnOrTst == "trn"
            end
            
            is_trn = (trnOrTst == "trn");

            if (spec == "mnistpy") || (spec == "mnistmat") || (spec == "fashion") || (spec == "emnistletters")
                if strcmp(spec, "mnistpy")
                    % clear classes
                    python_code = py.importlib.import_module("python_code");
                    py.importlib.reload(python_code);
                    if is_trn
                        if isnan(n_per_class)
                            x = struct(python_code.dataset(is_trn, int64(-1)));
                        else
                            assert(n_per_class <= 5421, "n_per_class must be <= 5421 (or NaN = all)");
                            x = struct(python_code.dataset(is_trn, int64(n_per_class)));
                        end
                    else
                        x = struct(python_code.dataset(is_trn, int64(-1)));
                    end
                    pixels = logical(x.data); % n_trn x 784 (comes in as uint8)
                    labelIdx = double(x.label_idx) + 1; % n_trn x 1 (comes in as uint8) indexes into uniq_classes
                    pixels = pixels';
                    img = reshape(pixels, 28, 28, []);
                    img = permute(img, [2,1,3]); % the data from python are transposed
                    img = reshape(img, 28*28, []);
                    dataset = struct();
                    dataset.sense = {img};
                    dataset.class_idx = labelIdx;
                    dataset.uniq_class = {'0','1','2','3','4','5','6','7','8','9'};
                elseif strcmp(spec, "mnistmat")
                    if is_trn
                        assert(isnan(n_per_class) || n_per_class <= 5421, 'n_per_class must be <= 5421 (or NaN = all)');
                        load(fullfile(Config.DATASET_DIR, 'img_captchas', 'mnist.trn-eqn-img-vec-noise.mat'), 'dataset'); % requires MatlabCommon
                    else
                        load(fullfile(Config.DATASET_DIR, 'img_captchas', 'mnist.tst-eqn-img-vec-noise.mat'), 'dataset'); % requires MatlabCommon
                    end
                elseif strcmp(spec, "fashion")
                    if is_trn
                        load(fullfile(Config.DATASET_DIR, 'img_captchas', 'fashionmnist.trn-eqn-img-vec-noise.mat'), 'dataset'); % requires MatlabCommon
                    else
                        load(fullfile(Config.DATASET_DIR, 'img_captchas', 'fashionmnist.tst-eqn-img-vec-noise.mat'), 'dataset'); % requires MatlabCommon
                    end
                elseif strcmp(spec, "emnistletters")
                    if is_trn
                        load(fullfile(Config.DATASET_DIR, 'img_captchas', 'emnist.byclasstrnlower-eqn-img-vec-noise.mat'), 'dataset'); % requires MatlabCommon
                    else
                        load(fullfile(Config.DATASET_DIR, 'img_captchas', 'emnist.byclasststlower-eqn-img-vec-noise.mat'), 'dataset'); % requires MatlabCommon
                    end
                end
                obj.pixels = dataset.sense{1};
                obj.label_idx = dataset.class_idx;
                obj.img_sz = [28,28,1];
                obj.uniq_classes = dataset.uniq_class;

                [row,col] = PixelRowCol(obj.img_sz);
                obj.node_name = cell(obj.n_nodes, 1);
                for j = 1 : obj.n_nodes
                    obj.node_name{j} = ['px_r',num2str(row(j)),'_c',num2str(col(j))];
                end
                
                if is_trn && ~strcmp(spec, "mnistpy")
                    idx = EqualizeN(obj.label_idx, n_per_class); % permanently reduce the number of images, while equalizing N
                else
                    idx = EqualizeN(obj.label_idx); % just equalize N
                end
                obj = obj.SubsetDatapoints(idx);
            elseif (spec == "ucicredit") || (spec == "ucicreditaustralian") || (spec == "ucicreditgerman")
                % load
                if spec == "ucicredit"
                    obj.meta = io.LoadCredit('uci_credit_screening', fullfile(Config.DATASET_DIR, 'credit', 'uci_credit_screening'));
                elseif spec == "ucicreditaustralian"
                    obj.meta = io.LoadCredit('uci_statlog_australian_credit', fullfile(Config.DATASET_DIR, 'credit', 'uci_statlog_australian_credit'));
                elseif spec == "ucicreditgerman"
                    obj.meta = io.LoadCredit('uci_statlog_german_credit', fullfile(Config.DATASET_DIR, 'credit', 'uci_statlog_german_credit'));
                    % continuous vars: a2_duration, a5_creditscore, a8_percent (uniques=1,2,3,4), a11_presentresidencesince (uniques=1,2,3,4), a13_age, 16_ncredits (uniques={1,2,3,4}), a18_ndependents (uniques=1,2)
                end

                % equalize n
                idx = EqualizeN(double(obj.meta.t.dv) + 1);
                obj.meta.t = obj.meta.t(idx,:);
                obj.meta.t_bin = obj.meta.t_bin(idx,:);

                % split trn from tst (keeping dv balanced)
                idx = RandSubsetDataset(double(obj.meta.t.dv) + 1, 0.5, RandStream('simdTwister', 'Seed', 77)); % must produce the same random sampling every time
                if is_trn
                    obj.meta.t = obj.meta.t(idx,:);
                    obj.meta.t_bin = obj.meta.t_bin(idx,:);
                else
                    obj.meta.t(idx,:) = [];
                    obj.meta.t_bin(idx,:) = [];
                end

                if spec == "ucicredit"
                    % separate the dv
                    obj.label_idx = double(obj.meta.t.dv) + 1;
                    obj.uniq_classes = {'-','+'}; % according to documentation, "1 = good, 2 = bad"
                    obj.meta.t_bin = removevars(obj.meta.t_bin, 'dv');
                    
                    % extract node info
                    logicalMsk = strcmp(varfun(@class, obj.meta.t_bin, 'OutputFormat', 'cell'), 'logical');
                    obj.node_name = obj.meta.t_bin.Properties.VariableNames(logicalMsk); % MUST be below the above lines
                    obj.pixels = table2array(obj.meta.t_bin(:,logicalMsk))'; % numel(obj.meta.pixel_names) x size(obj.meta.t, 1)
                    
                    % add binned versions of the non-logical fields
                    n_spatial_stops = 5;
                    nonlogicalIdx = find(~logicalMsk);
                    for j = 1 : numel(nonlogicalIdx) % all of these vars are sparse (rarely far from zero)
                        data = encode.TransformScalar2SpatialScalar(normalize(table2array(obj.meta.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
%                         data = encode.TransformScalar2SpikeViaThresh(sense, thresh);
                        data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
                        obj.pixels = cat(1, obj.pixels, data');
                        for k = 1 : n_spatial_stops
                            obj.node_name{end+1} = [obj.meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)},'_',num2str(k)];
                        end
                    end
                elseif spec == "ucicreditaustralian"
                    % separate the dv
                    obj.label_idx = double(obj.meta.t.dv) + 1;
                    obj.uniq_classes = {'-','+'}; % according to documentation, "1 = good, 2 = bad"
                    obj.meta.t_bin = removevars(obj.meta.t_bin, 'dv');
                    
                    % extract node info
                    logicalMsk = strcmp(varfun(@class, obj.meta.t_bin, 'OutputFormat', 'cell'), 'logical');
                    obj.node_name = obj.meta.t_bin.Properties.VariableNames(logicalMsk); % MUST be below the above lines
                    obj.pixels = table2array(obj.meta.t_bin(:,logicalMsk))'; % numel(obj.meta.pixel_names) x size(obj.meta.t, 1)
                    
                    % add binned versions of the non-logical fields
                    n_spatial_stops = 5;
                    nonlogicalIdx = find(~logicalMsk);
                    for j = 1 : numel(nonlogicalIdx) % all of these vars are sparse (rarely far from zero)
                        data = encode.TransformScalar2SpatialScalar(normalize(table2array(obj.meta.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
                        data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
                        obj.pixels = cat(1, obj.pixels, data');
                        for k = 1 : n_spatial_stops
                            obj.node_name{end+1} = [obj.meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)},'_',num2str(k)];
                        end
                    end
                elseif spec == "ucicreditgerman"
                    % separate the dv
                    obj.label_idx = double(obj.meta.t.dv) + 1;
                    obj.uniq_classes = {'good','bad'}; % according to documentation, "1 = good, 2 = bad"
                    obj.meta.t_bin = removevars(obj.meta.t_bin, 'dv');
                    
                    % extract node info
                    logicalMsk = strcmp(varfun(@class, obj.meta.t_bin, 'OutputFormat', 'cell'), 'logical');
                    obj.node_name = obj.meta.t_bin.Properties.VariableNames(logicalMsk); % MUST be below the above lines
                    obj.pixels = table2array(obj.meta.t_bin(:,logicalMsk))'; % numel(obj.meta.pixel_names) x size(obj.meta.t, 1)
                    
                    % add binned versions of the non-logical fields
                    n_spatial_stops = 5;
                    nonlogicalIdx = find(~logicalMsk);
                    for j = 1 : numel(nonlogicalIdx) % 1 and 2 are sparse
                        varName = obj.meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)};
                        if strcmp(varName, 'a8_percent') || strcmp(varName, 'a11_presentresidencesince') || strcmp(varName, '16_ncredits')
                            % all three vars hold integer values 1,2,3,4 - treat as categorical
                            for k = 1 : 4
                                obj.pixels = cat(1, obj.pixels, obj.meta.t_bin.(varName)' == k);
                                obj.node_name{end+1} = [varName,'_',num2str(k)];
                            end
                        elseif strcmp(varName, 'a18_ndependents') % takes integer values 1,2 - treat as binary
                            obj.meta.t_bin.a18_ndependents = (obj.meta.t_bin.a18_ndependents == 2);
                            obj.pixels = cat(1, obj.pixels, obj.meta.t_bin.a18_ndependents');
                            obj.node_name{end+1} = 'a18_ndependents';
                        else
                            data = encode.TransformScalar2SpatialScalar(normalize(table2array(obj.meta.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
                            data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
                            obj.pixels = cat(1, obj.pixels, data');
                            for k = 1 : n_spatial_stops
                                obj.node_name{end+1} = [obj.meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)},'_',num2str(k)];
                            end
                        end
                    end
                end
            elseif spec == "clevr"
                obj.img_sz = [80,120,24];
                obj.labeldata = CachedCompute(@LoadAndBinarizeCLEVR, is_trn);

                if is_trn
                    colors = cast([ ...
                        255,0,0;... % red
                        0,255,0;... % green
                        0,0,255;... % blue
                        255,255,0;... % yellow
                        255,0,255;... % magenta
                        0,255,255;... % cyan
                    ], 'uint8');
                    colors = reshape(colors, 1, size(colors, 1), size(colors, 2));
                    bar1 = 255 .* ones(obj.img_sz(1), 1, 3, 'uint8');
                    bar2 = 255 .* ones(1, 4*obj.img_sz(2) + 3, 3, 'uint8');
                    orange = cast([255,127,0], 'uint8');
                    for i = 1 : 20
                        rfoveated = round(min(obj.img_sz(2), max(1, obj.labeldata.objects{i}(1).pixel_coords(1))));
                        cfoveated = round(min(obj.img_sz(1), max(1, obj.labeldata.objects{i}(1).pixel_coords(2))));
                        img1 = cast([], 'uint8');
                        img2 = cat(2, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'), bar1, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'));
                        img3 = cat(2, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'), bar1, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'));
                        img4 = cat(2, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'), bar1, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'));
                        img5 = cat(2, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'), bar1, zeros(obj.img_sz(1), obj.img_sz(2), 3, 'uint8'));
                        for j = 1 : 6
                            l = colors(1,j,:) .* cast(repmat(obj.labeldata.img(:,:,j,i), 1, 1, 3), 'uint8'); % implicit expansion
                            r = colors(1,j,:) .* cast(repmat(obj.labeldata.img(:,:,6+j,i), 1, 1, 3), 'uint8'); % implicit expansion
                            t = colors(1,j,:) .* cast(repmat(obj.labeldata.img(:,:,12+j,i), 1, 1, 3), 'uint8'); % implicit expansion
                            b = colors(1,j,:) .* cast(repmat(obj.labeldata.img(:,:,18+j,i), 1, 1, 3), 'uint8'); % implicit expansion
                            img1 = cat(1, img1, cat(2, l, bar1, r, bar1, t, bar1, b), bar2);
                        
                            img2(:,1:obj.img_sz(2),:) = max(max(img2(:,1:obj.img_sz(2),:), l), r);
                            img2(:,obj.img_sz(2)+2:end,:) = max(max(img2(:,obj.img_sz(2)+2:end,:), t), b);
                            img3(:,1:obj.img_sz(2),:) = imtranslate(img2(:,1:obj.img_sz(2),:), [-cfoveated+obj.img_sz(1)/2,-rfoveated+obj.img_sz(2)/2,0]);
                            img3(:,obj.img_sz(2)+2:end,:) = imtranslate(img2(:,obj.img_sz(2)+2:end,:), [-cfoveated+obj.img_sz(1)/2,-rfoveated+obj.img_sz(2)/2,0]);
                        end
                        img1 = imresize(img1, 4, 'nearest');
                        img2 = imresize(img2, 4, 'nearest');
                        img3 = imresize(img3, 4, 'nearest');
                        img4 = imresize(img4, 4, 'nearest');
                        img5 = imresize(img5, 4, 'nearest');
                        for k1 = 1 : numel(obj.labeldata.objects{i})
                            r1 = round(obj.labeldata.objects{i}(k1).pixel_coords(1));
                            c1 = round(obj.labeldata.objects{i}(k1).pixel_coords(2));
                            for k2 = k1+1 : numel(obj.labeldata.objects{i})
                                r2 = obj.labeldata.objects{i}(k2).pixel_coords(1);
                                c2 = obj.labeldata.objects{i}(k2).pixel_coords(2);
                                img2 = insertShape(img2, 'line', 4.*[r1,c1,r2,c2], Color=orange, LineWidth=3, Opacity=1);
                                img2 = insertShape(img2, 'line', 4.*[obj.img_sz(2)+1+r1,c1,obj.img_sz(2)+1+r2,c2], Color=orange, LineWidth=3, Opacity=1);
                                img3 = insertShape(img3, 'line', 4.*[r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2,r2-rfoveated+obj.img_sz(2)/2,c2-cfoveated+obj.img_sz(1)/2], Color=orange, LineWidth=3, Opacity=1);
                                img3 = insertShape(img3, 'line', 4.*[obj.img_sz(2)+1+r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2,obj.img_sz(2)+1+r2-rfoveated+obj.img_sz(2)/2,c2-cfoveated+obj.img_sz(1)/2], Color=orange, LineWidth=3, Opacity=1);
                                img4 = insertShape(img4, 'line', 4.*[r1,c1,r2,c2], Color=orange, LineWidth=3, Opacity=1);
                                img4 = insertShape(img4, 'line', 4.*[obj.img_sz(2)+1+r1,c1,obj.img_sz(2)+1+r2,c2], Color=orange, LineWidth=3, Opacity=1);
                            end
                        end
                        for k1 = 2 : numel(obj.labeldata.objects{i})
                            r1 = round(obj.labeldata.objects{i}(k1).pixel_coords(1));
                            c1 = round(obj.labeldata.objects{i}(k1).pixel_coords(2));
                            if abs(r1 - rfoveated) > abs(c1 - cfoveated)
                                img5 = insertShape(img5, 'line', 4.*[r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2,obj.img_sz(2)/2,obj.img_sz(1)/2], Color=orange, LineWidth=3, Opacity=1);
                                img5 = insertText(img5, 4.*([r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2] + ([obj.img_sz(2)/2,obj.img_sz(1)/2]-[r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2])./2), '©M', 'BoxColor', orange, 'AnchorPoint', 'Center', 'FontSize', 16);
                            end
                            if abs(c1 - cfoveated) > abs(r1 - rfoveated)
                                img5 = insertShape(img5, 'line', 4.*[obj.img_sz(2)+1+r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2,obj.img_sz(2)+1+obj.img_sz(2)/2,obj.img_sz(1)/2], Color=orange, LineWidth=3, Opacity=1);
                                img5 = insertText(img5, 4.*([obj.img_sz(2)+1+r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2] + ([obj.img_sz(2)+1+obj.img_sz(2)/2,obj.img_sz(1)/2]-[obj.img_sz(2)+1+r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2])./2), '©N', 'BoxColor', orange, 'AnchorPoint', 'Center', 'FontSize', 16);
                            end
                        end
                        for k1 = 1 : numel(obj.labeldata.objects{i})
                            r1 = round(obj.labeldata.objects{i}(k1).pixel_coords(1));
                            c1 = round(obj.labeldata.objects{i}(k1).pixel_coords(2));
                            img4 = insertText(img4, 4.*[r1,c1], num2str(k1), 'TextColor', 'white', 'BoxColor', 'black', 'AnchorPoint', 'Center', 'FontSize', 16);
                            img4 = insertText(img4, 4.*[obj.img_sz(2)+1+r1,c1], num2str(k1), 'TextColor', 'white', 'BoxColor', 'black', 'AnchorPoint', 'Center', 'FontSize', 16);
                            
                            if k1 == 1 % foveated
                                symbol = '¢XX';
                            else
                                if r1 < rfoveated
                                    symbol = '¢AA'; % leftward
                                else
                                    symbol = '¢BB'; % rightward
                                end
                            end
                            if (k1 == 1) || (abs(r1 - rfoveated) > abs(c1 - cfoveated))
                                img5 = insertText(img5, 4.*[r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2], symbol, 'TextColor', 'white', 'BoxColor', 'black', 'AnchorPoint', 'Center', 'FontSize', 16);
                            end
                            if k1 == 1 % foveated
                                symbol = '¢XX';
                            else
                                if c1 < cfoveated
                                    symbol = '¢CC'; % above
                                else
                                    symbol = '¢DD'; % below
                                end
                            end
                            if (k1 == 1) || (abs(c1 - cfoveated) > abs(r1 - rfoveated))
                                img5 = insertText(img5, 4.*[obj.img_sz(2)+1+r1-rfoveated+obj.img_sz(2)/2,c1-cfoveated+obj.img_sz(1)/2], symbol, 'TextColor', 'white', 'BoxColor', 'black', 'AnchorPoint', 'Center', 'FontSize', 16);
                            end
                        end
                        imwrite(img1, ['./clevr_img',num2str(i),'_channels.png']);
                        imwrite(img2, ['./clevr_img',num2str(i),'_relations.png']);
                        imwrite(img3, ['./clevr_img',num2str(i),'_relationsfoveated.png']);
                        imwrite(img4, ['./clevr_img',num2str(i),'_numberedrelations.png']);
                        imwrite(img5, ['./clevr_img',num2str(i),'_numberedrelationsfoveated.png']);
                    end
                end
                error("clevr implementation incomplete")
            else
                error("unexpected frontend spec");
            end
        end


        function x = SubsetDatapoints(obj, keep)
            assert(islogical(keep) || all(IsIdx(keep))); % it's either a mask or an index
            x = obj; % make a copy
            x.pixels = x.pixels(:,keep);
            x.label_idx = x.label_idx(keep);
        end


        % gets
        function x = get.n_pts(obj)
            x = size(obj.pixels, 2);
        end
        function x = get.n_nodes(obj)
            x = size(obj.pixels, 1);
        end
        function x = get.n_classes(obj)
            x = numel(obj.uniq_classes);
        end
    end
end