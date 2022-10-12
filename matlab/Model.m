% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% set of ComponentBanks, with connections amonst them
% currently, we one-to-one connectivity (compbanks{1} is the input to compbanks{2} etc)
classdef Model
    properties
        g (1,1) digraph = digraph() % contains the connections amongst component banks
        compbanks (1,1) struct = struct() % each field a ComponentBank
    end
    properties (Dependent) % computed, derivative properties
        n_sense              % scalar (int-valued numeric) input dimensionality, e.g. number of pixels
        n_label              % scalar (int-valued numeric) number of label dimensions / items / classes
        n_compbanks          % scalar (int-valued numeric)
        input_bank_names     % n_compbanks x 1 (cellstr)
        output_bank_name     % (char)
        compbank_names       % n_compbanks x 1 (cellstr)
        tier1_compbank_names % ? x 1 (cellstr)
        encode_spec          % scalar (struct)
    end


    methods
        function obj = Model(layout, n_sense, n_label, nodeName) % constructor
            obj.g = addnode(obj.g, struct2table(struct(Name='sense', n_out=n_sense)));
            obj.g = addnode(obj.g, struct2table(struct(Name='label', n_out=n_label)));
            
            for i = 1 : numel(layout)
                if isfield(layout{i}, 'connec')
                    connecSpec = ParseList(layout{i}.connec);
                    for j = 1 : numel(connecSpec)
                        srcdst = strsplit(connecSpec{j}, '-->');
                        
                        warning('off', 'MATLAB:table:RowsAddedExistingVars'); % not a problem
                        obj.g = addedge(obj.g, srcdst{1}, srcdst{2});

                        assert(~strcmp(srcdst{2}, 'sense')); % reserved word
                        assert(indegree(obj.g, srcdst{2}) == 1); % currently, we only support one input per component bank (for simplicity, so that n_cmp of the input = n_nodes of the output)
                    end
                end
            end
            
            for i = 1 : numel(layout)
                if isfield(layout{i}, 'name')
                    [~,upstreamBanks] = inedges(obj.g, layout{i}.name);
                    obj.g.Nodes.encode_spec{strcmp(obj.g.Nodes.Name, layout{i}.name)} = layout{i}.encode_spec;
                    if any(strcmp(upstreamBanks, 'sense'))
                        obj.compbanks.(layout{i}.name) = ComponentBank(layout{i}.graph_type, layout{i}.edge_type_filter, nodeName);
                    else
                        obj.compbanks.(layout{i}.name) = ComponentBank(layout{i}.graph_type, layout{i}.edge_type_filter, {});
                    end
                end
            end
        end


        function [compCode,premergeIdx] = Encode(obj, dat)
            arguments
                obj, dat(1,1) Dataset
            end
            [compCode,premergeIdx] = EncodeHelper(obj, struct(sense=dat.pixels), struct(sense=[]), 'sense');
        end
        function [compCode,premergeIdx] = EncodeHelper(obj, compCode, premergeIdx, currSrc)
            [~,dstBanks] = outedges(obj.g, currSrc);
            for i = 1 : numel(dstBanks)
                if ~strcmp(dstBanks{i}, 'out')
                    assert(~isfield(compCode, dstBanks{i}), 'no support for graph cycles');
                    [compCode.(dstBanks{i}),premergeIdx.(dstBanks{i})] = Encode(obj.compbanks.(dstBanks{i}), compCode.(currSrc), obj.g.Nodes.encode_spec{findnode(obj.g, dstBanks{i})});
                end
            end
            for i = 1 : numel(dstBanks)
                if ~strcmp(dstBanks{i}, 'out')
                    [compCode,premergeIdx] = EncodeHelper(obj, compCode, premergeIdx, dstBanks{i}); % recurse
                end
            end
        end


        function idx = GroupIdx(obj, grouperBank)
            [~,groupedBank] = inedges(obj.g, grouperBank);
            assert(numel(groupedBank) == 1);
            groupedBank = groupedBank{1};
            
            [r,c] = find(obj.compbanks.(grouperBank).edge_states);
            assert(numel(r) == obj.compbanks.(groupedBank).n_cmp); % this function can only handle one-group-per-component
            idx = zeros(1, obj.compbanks.(groupedBank).n_cmp);
            idx(r) = c;
        end


        function [row,col] = PixelCoords(obj, bank, imgSz)
            row = zeros(1, obj.compbanks.(bank).n_cmp);
            col = zeros(1, obj.compbanks.(bank).n_cmp);
            if strcmp(bank, 'connectedpart')
                [pxRow,pxCol] = PixelRowCol(imgSz);
            else
                pxRow = [];
                pxCol = [];
                [~,srcBanks] = inedges(obj.g, bank);
                for i = 1 : numel(srcBanks)
                    [tempRow,tempCol] = PixelCoords(obj, srcBanks{i}, imgSz); % recurse
                    pxRow = cat(1, pxRow, tempRow(:));
                    pxCol = cat(1, pxCol, tempCol(:));
                end
            end
            for i = 1 : obj.compbanks.(bank).n_cmp
                edgeMsk = (obj.compbanks.(bank).edge_states(:,i) ~= EDG.NULL);
                pixelIdx = obj.didx(edgeMsk,:); % get the pixel indices associated with component i
                row(i) = mean(pxRow(pixelIdx(:)));
                col(i) = mean(pxCol(pixelIdx(:)));
            end
        end


        function obj = InsertComponents(obj, bank, n_new)
            obj.compbanks.(bank) = obj.compbanks.(bank).InsertComponents(n_new);
            
            % cleanup downstream
            % if changing components, must change downstream nodes
            [~,downstreamBanks] = outedges(obj.g, bank);
            for i = 1 : numel(downstreamBanks)
                str = downstreamBanks{i};
                if ~strcmp(str, 'out')
                    obj.compbanks.(str) = obj.compbanks.(str).InsertNodes(obj.compbanks.(bank).cmp_name(end-n_new+1:end));
                end
            end
        end


        function obj = SubsetComponents(obj, bank, keep)
            assert(islogical(keep) || all(IsIdx(keep))); % it's either a mask or an index
            
            obj.compbanks.(bank) = obj.compbanks.(bank).SubsetComponents(keep);
            
            % cleanup downstream
            % if changing components, must change downstream nodes
            [~,downstreamBanks] = outedges(obj.g, bank);
            for i = 1 : numel(downstreamBanks)
                str = downstreamBanks{i};
                if ~strcmp(str, 'out')
                    obj.compbanks.(str) = obj.compbanks.(str).RemoveNodes(obj.compbanks.(str).node_name(~keep));
                end
            end
        end


        function obj = ClearComponents(obj, bank)
            obj = SubsetComponents(obj, bank, false(obj.compbanks.(bank).n_cmp, 1));
        end


        function obj = Cleanup(obj)
            fn = fieldnames(obj.compbanks);
            for i = 1 : numel(fn)
                % remove any components that have no members/inputs
                keep = any(obj.compbanks.(fn{i}).edge_states, 1);
                obj = SubsetComponents(obj, fn{i}, keep);
            end
        end


        % gets
        function x = get.n_sense(obj)
            x = obj.g.Nodes.n_out(findnode(obj.g, 'sense'));
        end
        function x = get.n_label(obj)
            x = obj.g.Nodes.n_out(findnode(obj.g, 'label'));
        end
        function x = get.n_compbanks(obj)
            x = numel(fieldnames(obj.compbanks));
        end
        function x = get.output_bank_name(obj)
            [~,x] = inedges(obj.g, 'out');
            assert(numel(x) == 1);
            x = x{1}; % the input to "out" is our output unit
        end
        function x = get.compbank_names(obj)
            x = fieldnames(obj.compbanks);
        end
        function x = get.tier1_compbank_names(obj)
            [~,x] = outedges(obj.g, 'sense');
        end
        function x = get.encode_spec(obj)
            x = obj.g.Nodes.encode_spec;
        end
    end
end