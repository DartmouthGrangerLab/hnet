% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
function [pixels,label_idx,uniq_classes,pixel_metadata,label_metadata,other_metadata] = LoadCredit(spec, is_trn)
    arguments
        spec(1,1) string, is_trn(1,1) logical
    end
    pixel_metadata = struct();
    label_metadata = struct();
    other_metadata = struct();

    % load
    if spec == "ucicredit"
        other_metadata = io.LoadCredit('uci_credit_screening', char(fullfile(Config.DATASET_DIR, "credit", "uci_credit_screening")));
    elseif spec == "ucicreditaustralian"
        other_metadata = io.LoadCredit('uci_statlog_australian_credit', char(fullfile(Config.DATASET_DIR, "credit", "uci_statlog_australian_credit")));
    elseif spec == "ucicreditgerman"
        other_metadata = io.LoadCredit('uci_statlog_german_credit', char(fullfile(Config.DATASET_DIR, "credit", "uci_statlog_german_credit")));
        % continuous vars: a2_duration, a5_creditscore, a8_percent (uniques=1,2,3,4), a11_presentresidencesince (uniques=1,2,3,4), a13_age, 16_ncredits (uniques={1,2,3,4}), a18_ndependents (uniques=1,2)
    else
        error("unexpected spec");
    end

    % equalize n
    idx = EqualizeN(double(other_metadata.t.dv) + 1);
    other_metadata.t = other_metadata.t(idx,:);
    other_metadata.t_bin = other_metadata.t_bin(idx,:);

    % split trn from tst (keeping dv balanced)
    idx = RandSubsetDataset(double(other_metadata.t.dv) + 1, 0.5, RandStream("simdTwister", "Seed", 77)); % must produce the same random sampling every time
    if is_trn
        other_metadata.t = other_metadata.t(idx,:);
        other_metadata.t_bin = other_metadata.t_bin(idx,:);
    else
        other_metadata.t(idx,:) = [];
        other_metadata.t_bin(idx,:) = [];
    end

    % separate the dv
    label_idx = double(other_metadata.t.dv) + 1;
    other_metadata.t_bin = removevars(other_metadata.t_bin, "dv");
    if spec == "ucicredit"
        uniq_classes = {'-','+'}; % according to documentation, "1 = good, 2 = bad"
    elseif spec == "ucicreditaustralian"
        uniq_classes = {'-','+'}; % according to documentation, "1 = good, 2 = bad"
    elseif spec == "ucicreditgerman"
        uniq_classes = {'good','bad'}; % according to documentation, "1 = good, 2 = bad"
    else
        error("unexpected spec");
    end

    logicalMsk = strcmp(varfun(@class, other_metadata.t_bin, "OutputFormat", "cell"), "logical");

    % extract node info
    pixel_metadata.name = other_metadata.t_bin.Properties.VariableNames(logicalMsk)'; % MUST be below the above lines
    pixels = table2array(other_metadata.t_bin(:,logicalMsk))'; % numel(other_metadata.pixel_names) x size(other_metadata.t, 1)

    % add binned versions of the non-logical fields
    n_spatial_stops = 5;
    nonlogicalIdx = find(~logicalMsk);
    for j = 1 : numel(nonlogicalIdx) % all of these vars are sparse (rarely far from zero)
        varName = other_metadata.t_bin.Properties.VariableNames{nonlogicalIdx(j)};
        if (spec == "ucicredit") || (spec == "ucicreditaustralian")
            data = encode.TransformScalar2SpatialScalar(normalize(table2array(other_metadata.t_bin(:,nonlogicalIdx(j))), 'range'), n_spatial_stops);
            data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
            pixels = cat(1, pixels, data');
            for k = 1 : n_spatial_stops
                pixel_metadata.name{end+1} = [varName,'_',num2str(k)];
            end
        elseif spec == "ucicreditgerman"
            if varName == "a8_percent" || varName == "a11_presentresidencesince" || varName == "16_ncredits"
                % all three vars hold integer values 1,2,3,4 - treat as categorical
                for k = 1 : 4
                    pixels = cat(1, pixels, other_metadata.t_bin.(varName)' == k);
                    pixel_metadata.name{end+1} = [varName,'_',num2str(k)];
                end
            elseif varName == "a18_ndependents" % takes integer values 1,2 - treat as binary
                other_metadata.t_bin.a18_ndependents = (other_metadata.t_bin.a18_ndependents == 2);
                pixels = cat(1, pixels, other_metadata.t_bin.a18_ndependents');
                pixel_metadata.name{end+1} = 'a18_ndependents';
            else
                data = encode.TransformScalar2SpatialScalar(normalize(table2array(other_metadata.t_bin(:,nonlogicalIdx(j))), "range"), n_spatial_stops);
                data = encode.TransformScalar2SpikeViaKWTA(data, 1, 2, []);
                pixels = cat(1, pixels, data');
                for k = 1 : n_spatial_stops
                    pixel_metadata.name{end+1} = [varName,'_',num2str(k)];
                end
            end
        else
            error("unexpected spec");
        end
    end

    pixel_metadata.chanidx = ones(size(pixels, 1), 1);
end
