% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% all components within a bank share one graph
classdef ComponentBank
    properties % access is controlled via set functions
        edge_states  (:,:) EDG    % n_edges x n_cmp
        cmp_metadata (1,1) struct % metadata for each component (all fields are n_cmp x 1)
    end
    properties (SetAccess=private)
        graph_type       (1,1) GRF
        g                (1,1) DiGraph % keeps track of the edges within this bank
        edge_type_filter (:,1) EDG
        imgsz
    end
    properties (Dependent) % computed, derivative properties
        n_cmp            % scalar (int-valued numeric) number of components
        cmp_name         % n_cmp x 1 (cell array of chars)
        edge_endnodes    % n_edges x 2 (cellstr) like didx was, but contains string node names not numeric node idxs
        edge_endnode_idx % n_edges x 2 (int-valued numeric) numeric version of above
    end


    methods
        function obj = ComponentBank(graphType, edgeTypeFilter, n_nodes, node_metadata, imgsz) % constructor
            arguments
                graphType(1,1) GRF, edgeTypeFilter(:,1) EDG, n_nodes(1,1), node_metadata(1,1) struct, imgsz
            end
            if ~exist("imgsz", "var")
                imgsz = [];
            end
            obj.imgsz = imgsz;
            obj.graph_type = graphType;
            obj.edge_type_filter = edgeTypeFilter;

            [didx,isright,isdown,chanidx] = NeighborPairs(obj.graph_type, n_nodes, obj.imgsz);
            
            obj.g = DiGraph();
            assert(isfield(node_metadata, "name"));
            assert(isfield(node_metadata, "chanidx"));
            fn = fieldnames(node_metadata);
            for i = 1 : numel(fn)
                if iscell(node_metadata.(fn{i}))
                    obj.g.node_metadata.(fn{i}) = {};
                else
                    obj.g.node_metadata.(fn{i}) = [];
                end
            end
            obj.g.edge_metadata.is_right = logical([]);
            obj.g.edge_metadata.is_down = logical([]);

            obj.g = obj.g.AddNodes(1:n_nodes);
            if ~isempty(didx)
                obj.g = obj.g.AddEdges(didx(:,1), didx(:,2));
            end

            % populate graph metadata fields
            for i = 1 : numel(fn)
                obj.g.node_metadata.(fn{i})(:) = node_metadata.(fn{i});
            end
            if obj.graph_type == GRF.GRID2DMULTICHAN
                obj.g.node_metadata.chanidx(:) = chanidx;
            end
            obj.g.edge_metadata.is_right(:) = isright;
            obj.g.edge_metadata.is_down(:) = isdown;
            
            obj.edge_states = EDG(uint8.empty(obj.g.n_edges, 0));
        end


        function obj = InsertComponents(obj, n_new)
            if obj.g.n_edges == 0
                obj.edge_states = EDG(uint8.empty(0, obj.n_cmp+n_new)); % avoids an error
            else
                obj.edge_states(:,end+(1:n_new)) = EDG.NULL;
            end
        end


        function obj = SubsetComponents(obj, keep)
            assert(islogical(keep) || all(IsIdx(keep))); % it's either a mask or an index
            
            m = obj.cmp_metadata;
            fn = fieldnames(m);
            for i = 1 : numel(fn)
                m.(fn{i}) = m.(fn{i})(keep);
            end
            obj.edge_states = obj.edge_states(:,keep);
            obj.cmp_metadata = m;
        end


        function obj = InsertNodes(obj, nodeIDs, nodeName)
            arguments
                obj, nodeIDs, nodeName
            end
            n_new_nodes = numel(nodeIDs);
            n_orig_nodes = obj.g.n_nodes; % BEFORE any changes to obj
            
            [didx,isEdgeRight,isEdgeDown,nodeChan] = NeighborPairs(obj.graph_type, n_orig_nodes + n_new_nodes, obj.imgsz);
            
            % insert new nodes
            origNodes = obj.g.nodes;
            obj.g = obj.g.AddNodes(nodeIDs);
            if exist("nodeName", "var") && ~isempty(nodeName)
                obj.g.node_metadata.name(end-n_new_nodes+1:end) = nodeName;
            end
            if obj.graph_type == GRF.GRID2DMULTICHAN
                obj.g.node_metadata.chanidx(:) = nodeChan;
            end
            
            % find any new edges
            edges = obj.g.nodes(didx);
            if numel(origNodes) > 0
                assert(obj.graph_type ~= GRF.GRID2D && obj.graph_type ~= GRF.GRID2DMULTICHAN);
                drop = (ismember(edges(:,1), origNodes) & ismember(edges(:,2), origNodes)); % n_edges x 1 (logical)
                edges(drop,:) = [];
                isEdgeRight(drop,:) = [];
                isEdgeDown(drop,:) = [];
            end
            obj.g = obj.g.AddEdges(edges(:,1), edges(:,2));
            n_new_edges = size(edges, 1);
            if obj.n_cmp == 0
                obj.edge_states = EDG(uint8.empty(n_new_edges, 0));
            else
                obj.edge_states(end+(1:n_new_edges),:) = EDG.NULL;
            end
            obj.g.edge_metadata.is_right(end-n_new_edges+1:end) = isEdgeRight;
            obj.g.edge_metadata.is_down(end-n_new_edges+1:end) = isEdgeDown;
        end


        function obj = RemoveNodes(obj, nodeIDs2Remove)
            assert(isnumeric(nodeIDs2Remove));
            
            obj.g = obj.g.RemoveNodes(nodeIDs2Remove);
            
            % remove edges that used unkept nodes
            if obj.g.n_nodes == 0
                mask = false(obj.g.n_edges, 1);
            else
                mask = ~ismember(obj.g.edge_endnode_src, nodeName2Remove) & ~ismember(obj.g.edge_endnode_dst, nodeName2Remove); % n_edges x 1 (logical)
            end
            obj.edge_states = obj.edge_states(mask,:);
        end


        function x = ToMatlabDigraph(obj)
            x = obj.g.tograph();
        end


        % sets
        function obj = set.edge_states(obj, edgeStates)
            if (CallingFile() ~= "ComponentBank") && ~isempty(obj.edge_states) % ~isempty = important for loading class from file
                assert(size(edgeStates, 1) == size(obj.edge_states, 1), "use special setter functions if changing number of edges in component bank");
                assert(size(edgeStates, 2) == size(obj.edge_states, 2), "use special setter functions if changing number of components in bank");
            end
            obj.edge_states = edgeStates;
        end


        % gets
        function x = get.n_cmp(obj)
            x = size(obj.edge_states, 2);
        end
        function x = get.cmp_name(obj)
            x = cell(1, obj.n_cmp);
            for i = 1 : obj.n_cmp
                x{i} = num2str(i);
            end
        end
        function x = get.edge_endnodes(obj)
            x = cat(2, obj.g.edge_endnode_src, obj.g.edge_endnode_dst);
        end
        function x = get.edge_endnode_idx(obj)
            idx = (1:obj.g.n_nodes)';
            idx = idx(obj.g.nodes);
            x = cat(2, idx(obj.g.edge_endnode_src), idx(obj.g.edge_endnode_dst));
        end
    end
end