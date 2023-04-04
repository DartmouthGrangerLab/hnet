% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
function [pixels,label_idx,uniq_classes,meta,node_name] = LoadCredit(spec, is_trn)
    arguments
        spec(1,1) string, is_trn(1,1) logical
    end

    % load
    if spec == "ucicredit"
        meta = io.LoadCredit('uci_credit_screening', fullfile(Config.DATASET_DIR, 'credit', 'uci_credit_screening'));
    elseif spec == "ucicreditaustralian"
        meta = io.LoadCredit('uci_statlog_australian_credit', fullfile(Config.DATASET_DIR, 'credit', 'uci_statlog_australian_credit'));
    elseif spec == "ucicreditgerman"
        meta = io.LoadCredit('uci_statlog_german_credit', fullfile(Config.DATASET_DIR, 'credit', 'uci_statlog_german_credit'));
        % continuous vars: a2_duration, a5_creditscore, a8_percent (uniques=1,2,3,4), a11_presentresidencesince (uniques=1,2,3,4), a13_age, 16_ncredits (uniques={1,2,3,4}), a18_ndependents (uniques=1,2)
    end

    % equalize n
    idx = EqualizeN(double(meta.t.dv) + 1);
    meta.t = meta.t(idx,:);
    meta.t_bin = meta.t_bin(idx,:);

    % split trn from tst (keeping dv balanced)
    idx = RandSubsetDataset(double(meta.t.dv) + 1, 0.5, RandStream('simdTwister', 'Seed', 77)); % must produce the same random sampling every time
    if is_trn
        meta.t = meta.t(idx,:);
        meta.t_bin = meta.t_bin(idx,:);
    else
        meta.t(idx,:) = [];
        meta.t_bin(idx,:) = [];
    end

    if spec == "ucicredit"
        % separate the dv
        label_idx = double(meta.t.dv) + 1;
        uniq_classes = {'-','+'}; % according to documentation, "1 = good, 2 = bad"
        meta.t_bin = removevars(meta.t_bin, 'dv');
        
        % extract node info
        logicalMsk = strcmp(varfun(@class, meta.t_bin, 'OutputFormat', 'cell'), 'logical');
        node_name = meta.t_bin.Properties.VariableNames(logicalMsk); % MUST be below the above lines
        pixels = table2array(meta.t_bin(:,logicalMsk))'; % numel(obj.meta.pixel_names) x size(obj.meta.t, 1)
        
        % add binned versions of the non-logical fields
        n_spatial_stops = 5;
        nonlogicalIdx = find(~logicalMsk);
        for j = 1 : numel(nonlogicalIdx) % all of these vars are sparse (rarely far from zero)
            data = encode.TransformScalar2SpatialScalar(normalize(table2array(meta.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
            data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
            pixels = cat(1, pixels, data');
            for k = 1 : n_spatial_stops
                node_name{end+1} = [meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)},'_',num2str(k)];
            end
        end
    elseif spec == "ucicreditaustralian"
        % separate the dv
        label_idx = double(meta.t.dv) + 1;
        uniq_classes = {'-','+'}; % according to documentation, "1 = good, 2 = bad"
        meta.t_bin = removevars(meta.t_bin, 'dv');
        
        % extract node info
        logicalMsk = strcmp(varfun(@class, meta.t_bin, 'OutputFormat', 'cell'), 'logical');
        node_name = meta.t_bin.Properties.VariableNames(logicalMsk); % MUST be below the above lines
        pixels = table2array(meta.t_bin(:,logicalMsk))'; % numel(obj.meta.pixel_names) x size(obj.meta.t, 1)
        
        % add binned versions of the non-logical fields
        n_spatial_stops = 5;
        nonlogicalIdx = find(~logicalMsk);
        for j = 1 : numel(nonlogicalIdx) % all of these vars are sparse (rarely far from zero)
            data = encode.TransformScalar2SpatialScalar(normalize(table2array(meta.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
            data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
            pixels = cat(1, pixels, data');
            for k = 1 : n_spatial_stops
                node_name{end+1} = [meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)},'_',num2str(k)];
            end
        end
    elseif spec == "ucicreditgerman"
        % separate the dv
        label_idx = double(meta.t.dv) + 1;
        uniq_classes = {'good','bad'}; % according to documentation, "1 = good, 2 = bad"
        meta.t_bin = removevars(meta.t_bin, 'dv');
        
        % extract node info
        logicalMsk = strcmp(varfun(@class, meta.t_bin, 'OutputFormat', 'cell'), 'logical');
        node_name = meta.t_bin.Properties.VariableNames(logicalMsk); % MUST be below the above lines
        pixels = table2array(meta.t_bin(:,logicalMsk))'; % numel(obj.meta.pixel_names) x size(obj.meta.t, 1)
        
        % add binned versions of the non-logical fields
        n_spatial_stops = 5;
        nonlogicalIdx = find(~logicalMsk);
        for j = 1 : numel(nonlogicalIdx) % 1 and 2 are sparse
            varName = meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)};
            if strcmp(varName, 'a8_percent') || strcmp(varName, 'a11_presentresidencesince') || strcmp(varName, '16_ncredits')
                % all three vars hold integer values 1,2,3,4 - treat as categorical
                for k = 1 : 4
                    pixels = cat(1, pixels, meta.t_bin.(varName)' == k);
                    node_name{end+1} = [varName,'_',num2str(k)];
                end
            elseif strcmp(varName, 'a18_ndependents') % takes integer values 1,2 - treat as binary
                meta.t_bin.a18_ndependents = (meta.t_bin.a18_ndependents == 2);
                pixels = cat(1, pixels, meta.t_bin.a18_ndependents');
                node_name{end+1} = 'a18_ndependents';
            else
                data = encode.TransformScalar2SpatialScalar(normalize(table2array(meta.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
                data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
                pixels = cat(1, pixels, data');
                for k = 1 : n_spatial_stops
                    node_name{end+1} = [meta.t_bin.Properties.VariableNames{nonlogicalIdx(j)},'_',num2str(k)];
                end
            end
        end
    end
end