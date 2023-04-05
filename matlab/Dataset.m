% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
classdef Dataset
    properties (SetAccess=immutable)
        frontend_spec (1,:) char
        img_sz        (:,1) = [] % [n_rows,n_cols,n_chan] or empty if not an image dataset
        uniq_classes  (:,1) cell
    end
    properties
        pixels         (:,:)      % n_nodes x n 
        label_idx      (:,1)      % n x 1 or empty
        pixel_metadata (1,1) struct = struct() % each entry n_nodes x 1
        label_metadata (1,1) struct = struct()
        other_metadata (1,1) struct = struct()
    end
    properties (Dependent)
        n_pts     % scalar (int-valued numeric) number of datapoints / images
        n_classes % scalar (int-valued numeric)
        n_nodes   % scalar (int-valued numeric) aka number of features, aka number of pixels (n_rows*n_cols*n_chan)
    end


    methods
        function obj = Dataset(frontendSpec, trnOrTst) % constructor
            arguments
                frontendSpec, trnOrTst(1,1) string
            end
            obj.frontend_spec = char(frontendSpec);
            spec = frontendSpec;
            if contains(frontendSpec, ".")
                temp = strsplit(frontendSpec, ".");
                spec = temp{1};
                n_per_class = Str2Double(temp{2}); % only used if trnOrTst == "trn"
            end
            
            is_trn = (trnOrTst == "trn");

            if (spec == "mnistpy") || (spec == "mnistmat") || (spec == "fashion") || (spec == "emnistletters")
                obj.img_sz = [28,28,1];
                [obj.pixels,obj.label_idx,obj.uniq_classes,obj.pixel_metadata,obj.label_metadata,obj.other_metadata] = LoadMNIST(spec, is_trn, n_per_class);
            elseif (spec == "ucicredit") || (spec == "ucicreditaustralian") || (spec == "ucicreditgerman")
                [obj.pixels,obj.label_idx,obj.uniq_classes,obj.pixel_metadata,obj.label_metadata,obj.other_metadata] = LoadCredit(spec, is_trn);
            elseif spec == "clevr"
                obj.img_sz = [80,120,24];
                [obj.pixels,obj.pixel_metadata,obj.label_metadata,obj.other_metadata] = LoadCLEVR(spec, is_trn);

                if is_trn
                    RenderCLEVRDatasetSummary(obj); % must be before the foveation
                end
                
                [row,col] = PixelRowCol(obj.img_sz);
                n_nodes = obj.img_sz(1) * obj.img_sz(2) * obj.img_sz(3);

                % to help with foveation, find connected parts in each image
                compbank = ComponentBank(GRF.GRID2DMULTICHAN, [EDG.NCONV,EDG.NIMPL], n_nodes, obj.pixel_metadata, obj.img_sz);
                compbank.cmp_metadata = struct(src_img_idx=[], src_chan=[]);
                for i = 1 : numel(obj.other_metadata.chan_color) % for each channel
                    compbank = compbank.InsertComponents(obj.other_metadata.n);
                    pixels = obj.other_metadata.img; % n_rows x n_cols x n_chan x n
                    pixels(:,:,[1:i-1,i+1:end],:) = 0; % mask all channels but one
                    pixels = reshape(pixels, [], obj.other_metadata.n); % n_nodes x n
                    compbank.edge_states(:,end-obj.other_metadata.n+1:end) = GetEdgeStates(pixels, compbank.edge_endnode_idx, compbank.edge_type_filter); % convert from pixels to edges
                    compbank.cmp_metadata.src_chan(end+(1:obj.other_metadata.n)) = i; % implicit expansion
                end
                compbank.cmp_metadata.src_img_idx = repmat((1:obj.other_metadata.n)', [obj.img_sz(3), 1]);
                max_length = 25; % max length of a connected component (e.g. Inf, 20)
                [newRelations,metadata] = ExtractConnectedPartComponents(compbank, obj.img_sz, max_length, 1.5);
                compbank = compbank.SubsetComponents(false(compbank.n_cmp, 1));
                compbank = compbank.InsertComponents(size(newRelations, 2));
                compbank.edge_states(:) = newRelations;
                partimgidx = metadata.src_img_idx;
                partpxcoords = zeros(compbank.n_cmp, 2); % (:,1) = row, (:,2) = col
                for i = 1 : compbank.n_cmp % borrowed from Model.PixelCoords()
                    edgeMsk = (compbank.edge_states(:,i) ~= EDG.NULL);
                    pixelIdx = compbank.edge_endnode_idx(edgeMsk,:); % get the pixel indices associated with component i
                    partpxcoords(i,1) = mean(row(pixelIdx(:)));
                    partpxcoords(i,2) = mean(col(pixelIdx(:)));
                end
                partpxcoords = round(partpxcoords);
                shiftr = obj.img_sz(1)/2 - partpxcoords(:,1);
                shiftc = obj.img_sz(2)/2 - partpxcoords(:,2);
                
                % foveate
                obj.other_metadata.n = numel(partimgidx);
                obj.label_metadata.image_idx = obj.label_metadata.image_idx(partimgidx);
                obj.other_metadata.objects   = obj.other_metadata.objects(partimgidx);
                obj.other_metadata.img       = obj.other_metadata.img(:,:,:,partimgidx);
                temp = zeros(size(obj.other_metadata.img), "like", obj.other_metadata.img);
                for i = 1 : obj.other_metadata.n
                    if shiftr(i) > 0
                        temp(shiftr(i):end,:,:,i) = obj.other_metadata.img(1:end-shiftr(i)+1,:,:,i);
                    elseif shiftr(i) < 0
                        temp(1:end+shiftr(i)+1,:,:,i) = obj.other_metadata.img(-shiftr(i):end,:,:,i);
                    end
                    if shiftc(i) > 0
                        temp(:,shiftc(i):end,:,i) = obj.other_metadata.img(:,1:end-shiftc(i)+1,:,i);
                    elseif shiftc(i) < 0
                        temp(:,1:end+shiftc(i)+1,:,i) = obj.other_metadata.img(:,-shiftc(i):end,:,i);
                    end
                    for j = 1 : numel(obj.other_metadata.objects{i}) % for each object
                        obj.other_metadata.objects{i}(j).pixel_coords(1:2) = obj.other_metadata.objects{i}(j).pixel_coords(1:2) - partpxcoords(i,:)';
                    end
                end
                obj.other_metadata.img = temp;
                obj.other_metadata.foveated_chan = metadata.src_chan;
                % re-set obj.pixels now that other_metadata.img has changed
                obj.pixels = reshape(obj.other_metadata.img, [], obj.other_metadata.n); % n_nodes*n_chan x n
            elseif spec == "clevrpossimple" % clevr positions, simple version
                % images are 320x240, but clevrpos isn't a standard image-based dataset so we don't set obj.img_sz
                [obj.pixels,obj.pixel_metadata,obj.label_metadata,obj.other_metadata] = LoadCLEVRPos(spec, is_trn);
            else
                error("unexpected frontend spec");
            end
        end


        function x = SubsetDatapoints(obj, keep)
            assert(islogical(keep) || all(IsIdx(keep))); % it's either a mask or an index
           
            x = obj; % make a copy
            
            fn = fieldnames(x.label_metadata);
            for i = 1 : numel(fn)
                if numel(x.label_metadata.(fn{i})) == obj.n_pts % a vector of correct length
                    x.label_metadata.(fn{i}) = x.label_metadata.(fn{i})(keep);
                end
                if size(x.label_metadata.(fn{i}), 2) == obj.n_pts % a matrix of correct length
                    x.label_metadata.(fn{i}) = x.label_metadata.(fn{i})(:,keep);
                end
            end

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