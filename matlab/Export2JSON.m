% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% dataset format: {
%     "comment": "<text notes>",
%     "name": "<name of dataset>",
%     "split": "<trn | tst>",
%     "data": [<list of numbers in which the outermost (leftmost) dimension is the number of datapoints>],
%     "label": [<if the data comes with class labels, one per datapoint, those are here, else empty list>],
%     <other metadata, depending on dataset>
% }
% model format: {
%     "comment": "<text notes>",
%     "links": "sense-->0,0-->1,1-->out", <where numbers index into the layout list>
%     "layout": [
%         {
%             "name": "<arbitrary component bank name>",
%             "h": [<list of numbers, n_cmp x n_nodes x n_nodes or empty list>],
%             "k": [<list of numbers, n_cmp x 1 or empty list>],
%             "learned_edge_states": [<list of numbers, ? x ? or empty list>],
%             "edge_endnode_idx": [...],
%             "edge_type_filter": [...],
%             "nonlinearity_mode": "<...>",
%             "n_winners": <int, ignored unless the layout calls for KWTA nonlinearity>
%         }, { ... }, ...
%     ]
% }
function [] = Export2JSON(trndat, tstdat, model, frontend_spec, model_name)
    arguments
        trndat (1,1) Dataset
        tstdat (1,1) Dataset
        model (1,1) Model
        frontend_spec (1,1) string
        model_name (1,1) string
    end

    % export trn dataset
    s = struct();
    s.comment = "";
    s.name = trndat.frontend_spec;
    s.split = "trn";
    s.data = trndat.pixels'; % now n x n_nodes
    s.label_idx = trndat.label_idx;
    txt = jsonencode(s);
    writelines(txt, fullfile(Config.OUT_DIR, frontend_spec + "_trn.dataset.json"));

    % export tst dataset
    s = struct();
    s.comment = "";
    s.name = tstdat.frontend_spec;
    s.split = "tst";
    s.data = tstdat.pixels'; % now n x n_nodes
    s.label_idx = tstdat.label_idx;
    txt = jsonencode(s);
    writelines(txt, fullfile(Config.OUT_DIR, frontend_spec + "_tst.dataset.json"));

    % export model
    s = struct();
    s.comment = "from Export2JSON.m";
    s.links = model.connec;
    s.layout = {};
    for i = 1 : model.n_compbanks
        compbank = model.compbanks.(model.compbank_names{i});
        encode_spec = model.g.Nodes.encode_spec{findnode(model.g, model.compbank_names{i})};
        H = cell(compbank.n_cmp, 1);
        k = zeros(compbank.n_cmp, 1);
        for j = 1 : compbank.n_cmp
            [H{j},k(j)] = GenerateCompositeH(compbank, j);
        end
        curr_bank = struct();
        curr_bank.name = model.compbank_names{i};
        curr_bank.h = zeros(numel(H), size(H{1}, 1), size(H{1}, 2), "int16");
        for j = 1 : numel(H)
            curr_bank.h(j,:,:) = full(H{j});
        end
        curr_bank.k = k;
        curr_bank.learned_edge_states = uint8(compbank.edge_states)';
        curr_bank.edge_endnode_idx = compbank.edge_endnode_idx - 1; % convert to 0-based indexing
        curr_bank.edge_type_filter = uint8(compbank.edge_type_filter);
        if encode_spec == "energy"
            curr_bank.nonlinearity_mode = "none";
            curr_bank.n_winners = 0;
        elseif encode_spec == "energy-->max"
            curr_bank.nonlinearity_mode = "max";
            curr_bank.n_winners = 0;
        elseif encode_spec == "energy-->maxabs"
            curr_bank.nonlinearity_mode = "maxabs";
            curr_bank.n_winners = 0;
        elseif startsWith(encode_spec, "energy-->wta")
            curr_bank.nonlinearity_mode = "wta";
            temp = strsplit(encode_spec, ".");
            curr_bank.n_winners = str2num(temp{2});
        elseif encode_spec == "energy-->nonzero"
            curr_bank.nonlinearity_mode = "nonzero";
            curr_bank.n_winners = 0;
        else
            error("unexpected encode_spec");
        end
        s.layout{end+1} = curr_bank;
    end
    
    txt = jsonencode(s);
    writelines(txt, fullfile(Config.OUT_DIR, model_name + ".hnetmodel.json"));
end