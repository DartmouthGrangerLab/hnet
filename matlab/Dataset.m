% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
classdef Dataset
    properties (SetAccess=immutable)
        frontend_spec (1,:) char
        img_sz        (1,:) = [] % [n_rows,n_cols,n_chan] or empty if not an image dataset
        uniq_classes  (1,:) cell
    end
    properties
        pixels    (:,:)      % n_nodes x n
        node_name (1,:) cell % n_nodes x 1
        label_idx (1,:)      % n x 1
        meta      (1,1) struct = struct()
    end
    properties (Dependent)
        n_pts     % scalar (int-valued numeric) number of datapoints / images
        n_classes % scalar (int-valued numeric) 
        n_nodes   % scalar (int-valued numeric) aka number of features, aka number of pixels (n_rows*n_cols*n_chan)
    end


    methods
        function obj = Dataset(frontendSpec, trnOrTst) % constructor
            obj.frontend_spec = frontendSpec;
            spec = frontendSpec;
            if any(frontendSpec == '.')
                temp = strsplit(frontendSpec, '.');
                spec = temp{1};
                n_per_class = Str2Double(temp{2}); % only used if trnOrTst = 'trn'
            end
            
            is_trn = false;
            if strcmp(trnOrTst, 'trn')
                is_trn = true;
            end

            if strcmp(spec, 'mnistpy')
                % clear classes
                python_code = py.importlib.import_module('python_code');
                py.importlib.reload(python_code);
                if is_trn
                    if isnan(n_per_class)
                        x = struct(python_code.dataset(is_trn, int64(-1)));
                    else
                        assert(n_per_class <= 5421, 'n_per_class must be <= 5421 (or NaN = all)');
                        x = struct(python_code.dataset(is_trn, int64(n_per_class)));
                    end
                else
                    x = struct(python_code.dataset(is_trn, int64(-1)));
                end
                pixels = logical(x.data); % n_trn x 784 (comes in as uint8)
                labelIdx = double(x.label_idx) + 1; % 1 x n_trn (comes in as uint8) indexes into uniq_classes
                pixels = pixels';
                img = reshape(pixels, 28, 28, []);
                img = permute(img, [2,1,3]); % the data from python are transposed
                img = reshape(img, 28*28, []);
                dataset = struct();
                dataset.sense = {img};
                dataset.class_idx = labelIdx;
                dataset.uniq_class = {'0','1','2','3','4','5','6','7','8','9'};
                
                obj.pixels = dataset.sense{1};
                obj.label_idx = dataset.class_idx;
                obj.img_sz = [28,28,1];
                obj.uniq_classes = dataset.uniq_class;

                [row,col] = PixelRowCol(obj.img_sz);
                obj.node_name = cell(obj.n_nodes, 1);
                for j = 1 : obj.n_nodes
                    obj.node_name{j} = ['px_r',num2str(row(j)),'_c',num2str(col(j))];
                end
                
                if is_trn && ~strcmp(spec, 'mnistpy')
                    idx = EqualizeN(obj.label_idx, n_per_class); % permanently reduce the number of images, while equalizing N
                else
                    idx = EqualizeN(obj.label_idx); % just equalize N
                end
                obj = obj.SubsetDatapoints(idx);
            elseif strcmp(spec, 'ucicreditgerman')
                % load
                obj.meta = io.LoadCredit('uci_statlog_german_credit', fullfile('..', 'datasets', 'credit', 'uci_statlog_german_credit'));
                % continuous vars: a2_duration, a5_creditscore, a8_percent (uniques=1,2,3,4), a11_presentresidencesince (uniques=1,2,3,4), a13_age, 16_ncredits (uniques={1,2,3,4}), a18_ndependents (uniques=1,2)
                
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
            else
                error('unexpected frontend spec');
            end
        end


        function x = SubsetDatapoints(obj, keep)
            assert(islogical(keep) || all(IsIdx(keep))); % it's either a mask or an index
            x = obj; % make a copy
            x.pixels = x.pixels(:,keep);
            x.label_idx = x.label_idx(:,keep);
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