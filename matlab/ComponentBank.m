% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite ___.
% all components within a bank share one graph
classdef ComponentBank
    properties % access is controlled via set functions
        edge_states (:,:) EDG    % n_edges x n_cmp
        meta        (1,1) struct % metadata for each component (all fields are 1 x n_cmp)
    end
    properties (SetAccess=private)
        graph_type       (1,1) GRF
        g                (1,1) digraph % keeps track of the edges within this bank
        edge_type_filter (:,1) EDG
    end
    properties (SetAccess=private, Transient=true)
        cache (1,1) struct
    end
    properties (Dependent) % computed, derivative properties
        n_cmp            % scalar (int-valued numeric) number of components
        n_nodes          % scalar (int-valued numeric) number of nodes
        n_edges          % scalar (int-valued numeric) number of edges in the graph
        node_name        % n_nodes x 1 (cell array of chars)
        cmp_name         % n_cmp x 1 (cell array of chars)
        edge_endnodes    % n_edges x 2 (numeric index into node_name) like didx was, but contains string node names not numeric node idxs
        edge_endnode_idx % n_edges x 2 (int-valued numeric) numeric version of above
    end


    methods
        function obj = ComponentBank(graphType, edgeTypeFilter, nodeName) % constructor
            obj.graph_type = graphType;
            obj.edge_type_filter = edgeTypeFilter;
            
            obj.g = digraph();
            obj.g = addnode(obj.g, nodeName);
            didx = NeighborPairs(obj.graph_type, obj.n_nodes);
            obj.g = addedge(obj.g, nodeName(didx(:,1)), nodeName(didx(:,2)));
            
            obj.edge_states = EDG(uint8.empty(size(didx, 1), 0));
        end


        function obj = InsertComponents(obj, n_new)
            if obj.n_edges == 0
                obj.edge_states = EDG(uint8.empty(0, obj.n_cmp+n_new)); % avoids an error
            else
                obj.edge_states(:,end+(1:n_new)) = EDG.NULL;
            end

            obj.cache = struct(); % clear cache
        end


        function obj = SubsetComponents(obj, keep)
            assert(islogical(keep) || all(IsIdx(keep))); % it's either a mask or an index
            
            metadata = obj.meta;
            fn = fieldnames(metadata);
            for i = 1 : numel(fn)
                metadata.(fn{i}) = metadata.(fn{i})(keep);
            end
            obj.edge_states = obj.edge_states(:,keep);
            obj.meta = metadata;

            obj.cache = struct(); % clear cache
        end


        function obj = InsertNodes(obj, nodeName)
            n_new_nodes = numel(nodeName);
            n_orig_nodes = obj.n_nodes; % BEFORE any changes to obj
            
            % insert new nodes
            origNodes = obj.node_name;
            obj.g = addnode(obj.g, nodeName);
            
            % find any new edges
            edges = obj.node_name(NeighborPairs(obj.graph_type, n_orig_nodes + n_new_nodes));
            if numel(origNodes) > 0
                assert(obj.graph_type ~= GRF.GRID2DSQR);
                drop = (ismember(edges(:,1), origNodes) & ismember(edges(:,2), origNodes)); % n_edges x 1 (logical)
                edges(drop,:) = [];
            end
            obj.g = addedge(obj.g, edges(:,1), edges(:,2));
            if obj.n_cmp == 0
                obj.edge_states = EDG(uint8.empty(size(edges, 1), 0));
            else
                obj.edge_states(end+(1:size(edges, 1)),:) = EDG.NULL;
            end

            obj.cache = struct(); % clear cache
        end


        function obj = RemoveNodes(obj, nodeName2Remove)
            assert(ischar(nodeName2Remove) || iscell(nodeName2Remove));
            
            % remove nodes
            obj.g = rmnode(obj.g, nodeName2Remove);
            
            % remove edges that used unkept nodes
            if obj.g.numnodes == 0
                mask = false(obj.n_edges, 1);
            else
                mask = ~ismember(obj.edges(:,1), nodeName2Remove) & ~ismember(obj.edges(:,2), nodeName2Remove); % n_edges x 1 (logical)
            end
            obj.edge_states = obj.edge_states(mask,:);

            obj.cache = struct(); % clear cache
        end


        % sets
        function obj = set.edge_states(obj, edgeStates)
            if ~strcmp(CallingFile(), 'ComponentBank') && ~isempty(obj.edge_states) % ~isempty = important for loading class from file
                assert(size(edgeStates, 1) == size(obj.edge_states, 1), 'use special setter functions if changing number of edges in component bank');
                assert(size(edgeStates, 2) == size(obj.edge_states, 2), 'use special setter functions if changing number of components in bank');
            end
            obj.edge_states = edgeStates;

            obj.cache = struct(); % clear cache
        end
        function obj = set.node_name(obj, nodeName)
            if ~strcmp(CallingFile(), 'ComponentBank') && ~isempty(obj.node_name) % ~isempty = important for loading class from file
                assert(numel(nodeName) == numel(obj.node_name), 'use InsertNodes() if changing number of nodes in component bank');
            end
            obj.node_name = nodeName;
        end
        function obj = set.meta(obj, meta)
            fn = fieldnames(meta);
            for i = 2 : numel(fn)
                assert(numel(meta.(fn{i})) == numel(meta.(fn{1})));
            end
            
            obj.meta = meta;
        end


        % gets
        function x = get.n_nodes(obj)
            x = obj.g.numnodes;
        end
        function x = get.n_edges(obj)
            x = obj.g.numedges; % aka size(obj.edge_states, 1)
        end
        function x = get.n_cmp(obj)
            x = size(obj.edge_states, 2);
        end
        function x = get.node_name(obj)
            x = obj.g.Nodes.Name;
        end
        function x = get.cmp_name(obj)
            x = cell(1, obj.n_cmp);
            for i = 1 : obj.n_cmp
                x{i} = num2str(i);
            end
        end
        function x = get.edge_endnodes(obj)
            x = obj.g.Edges.EndNodes;
        end
        function x = get.edge_endnode_idx(obj)
            if ~isfield(obj.cache, 'edge_endnode_idx') % below code can get super slow
                nodeName = table2cell(obj.g.Nodes);
                edges = obj.g.Edges.EndNodes;
                obj.cache.edge_endnode_idx = zeros(obj.g.numedges, 2);
                for i = 1 : obj.g.numedges
                    obj.cache.edge_endnode_idx(i,1) = find(ismember(nodeName, edges(i,1)));
                    obj.cache.edge_endnode_idx(i,2) = find(ismember(nodeName, edges(i,2)));
                end
                % confirmed all(strcmp(nodeName(x), edges))
            end
            x = obj.cache.edge_endnode_idx;
        end
    end
end