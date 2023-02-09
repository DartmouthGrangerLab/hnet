% directed or undirected graph
% more feature-rich and simpler to work with than matlab's built in graph & digraph
classdef DiGraph
    properties (SetAccess=private)
        nodes            (:,1) % n_nodes x 1 (numeric) ID of each node
        edge_endnode_src (:,1) % n_edges x 1 (numeric) ID (not index) of edge start node
        edge_endnode_dst (:,1) % n_edges x 1 (numeric) ID (not index) of edge end node
    end
    properties
        node_metadata (1,1) struct
        edge_metadata (1,1) struct
    end
    properties (Dependent) % computed, derivative properties
        n_nodes % scalar (int-valued numeric) number of nodes
        n_edges % scalar (int-valued numeric) number of edges in the graph
        edge_endnode_src_idx
        edge_endnode_dst_idx
    end
    
    
    methods
        function obj = DiGraph() % constructor
            
        end
        
        
        function obj = AddNodes(obj, nodeid)
            assert(~any(any(obj.nodes(:) == nodeid(:)')));
            
            obj.nodes = cat(1, obj.nodes, nodeid(:));
            
            fn = fieldnames(obj.node_metadata);
            for i = 1 : numel(fn)
                if iscell(obj.node_metadata.(fn{i}))
                    obj.node_metadata.(fn{i}) = cat(1, obj.node_metadata.(fn{i}), cell(numel(nodeid), 1));
                else
                    obj.node_metadata.(fn{i}) = cat(1, obj.node_metadata.(fn{i}), zeros(numel(nodeid), 1, 'like', obj.node_metadata.(fn{i})));
                end
            end
            
            obj.Validate();
        end
        
        
        function obj = AddEdges(obj, srcid, dstid)
            assert(numel(srcid) == numel(dstid));
            for i = 1 : numel(srcid)
                assert(any(srcid(i) == obj.nodes));
                assert(any(dstid(i) == obj.nodes));
            end
            
            obj.edge_endnode_src = cat(1, obj.edge_endnode_src, srcid(:));
            obj.edge_endnode_dst = cat(1, obj.edge_endnode_dst, dstid(:));

            fn = fieldnames(obj.edge_metadata);
            for i = 1 : numel(fn)
                if iscell(obj.edge_metadata.(fn{i}))
                    obj.edge_metadata.(fn{i}) = cat(1, obj.edge_metadata.(fn{i}), cell(numel(srcid), 1));
                else
                    obj.edge_metadata.(fn{i}) = cat(1, obj.edge_metadata.(fn{i}), zeros(numel(srcid), 1, 'like', obj.edge_metadata.(fn{i})));
                end
            end
            
            obj.Validate();
        end
        
        
        function obj = RemoveNodes(obj, node_drop_ids)
            drop = ismember(obj.edge_endnode_src, node_drop_ids) | ismember(obj.edge_endnode_dst, node_drop_ids); % n_edges x 1 (logical)
            obj = obj.RemoveEdges(drop);

            drop = false(obj.n_nodes, 1);
            for i = 1 : numel(node_drop_ids)
                drop = drop | (obj.nodes == node_drop_ids(i));
            end
            obj.nodes(drop) = [];

            fn = fieldnames(obj.node_metadata);
            for i = 1 : numel(fn)
                obj.node_metadata.(fn{i})(drop) = [];
            end

            obj.Validate();
        end
        
        
        function obj = RemoveEdges(obj, edge_drop)
            obj.edge_endnode_src(edge_drop) = [];
            obj.edge_endnode_dst(edge_drop) = [];
            
            fn = fieldnames(obj.edge_metadata);
            for i = 1 : numel(fn)
                obj.edge_metadata.(fn{i})(edge_drop) = [];
            end
        end
        
        
        function [link_idx,nid] = inlinks(obj, node)
            assert(isscalar(node) && any(node == self.nodes));
            link_idx = obj.FindEdges([], node);
            nid = obj.edge_endnode_src(link_idx);
        end

        
        function [link_idx,nid] = outlinks(obj, node)
            assert(isscalar(node) && any(node == self.nodes));
            link_idx = obj.FindEdges(node, []);
            nid = obj.edge_endnode_dst(link_idx);
        end

        
        function x = indegree(obj, nodes)
            if ~exist('nodes', 'var') || isempty(nodes)
                nodes = obj.nodes;
            end
            x = zeros(numel(nodes), 1);
            for i = 1 : numel(nodes)
                x(i) = sum(obj.edge_endnode_dst == nodes(i));
            end
        end

        
        function x = outdegree(obj, nodes)
            if ~exist('nodes', 'var') || isempty(nodes)
                nodes = obj.nodes;
            end
            x = zeros(numel(nodes), 1);
            for i = 1 : numel(nodes)
                x(i) = sum(obj.edge_endnode_src == nodes(i));
            end
        end
        
        
        function x = FindEdges(obj, srcnode, dstnode)
            assert(numel(srcnode) < 2 && numel(dstnode) < 2);
            if ~isempty(srcnode) && ~isempty(dstnode)
                x = find((obj.edge_endnode_src == srcnode) & (obj.edge_endnode_dst == dstnode));
            elseif ~isempty(srcnode)
                x = find(self.edge_endnode_src == srcnode);
            elseif ~isempty(dstnode)
                x = find(self.edge_endnode_dst == dstnode);
            else
                error('srcnode and dstnode cannot both be empty');
            end
        end


        function x = tograph(obj)
            x = digraph();
            if isfield(obj.node_metadata, 'name')
                x = addnode(x, obj.node_metadata.name);
                x = addedge(x, obj.node_metadata.name(obj.edge_endnode_src), obj.node_metadata.name(obj.edge_endnode_dst));
            else
                x = addnode(x, obj.nodes);
                x = addedge(x, obj.nodes(obj.edge_endnode_src), obj.nodes(obj.edge_endnode_dst));
            end
        end
        
        
        % gets
        function x = get.n_nodes(obj)
            x = numel(obj.nodes);
        end
        function x = get.n_edges(obj)
            x = numel(obj.edge_endnode_src);
        end
        function x = get.edge_endnode_src_idx(obj)
            x = zeros(obj.n_edges, 1);
            for i = 1 : obj.n_edges
                x(i) = find(obj.edge_endnode_src(i) == obj.nodes);
            end
        end
        function x = get.edge_endnode_dst_idx(obj)
            x = zeros(obj.n_edges, 1);
            for i = 1 : obj.n_edges
                x(i) = find(obj.edge_endnode_dst(i) == obj.nodes);
            end
        end
        
        
        function Validate(obj)
            assert(numel(obj.edge_endnode_src) == numel(obj.edge_endnode_dst));
            assert(all(ismember(obj.edge_endnode_src, obj.nodes)));
            assert(all(ismember(obj.edge_endnode_dst, obj.nodes)));
            fn = fieldnames(obj.node_metadata);
            for i = 1 : numel(fn)
                assert(numel(obj.node_metadata.(fn{i})) == obj.n_nodes);
            end
            fn = fieldnames(obj.edge_metadata);
            for i = 1 : numel(fn)
                assert(numel(obj.edge_metadata.(fn{i})) == obj.n_edges);
            end
        end
    end
end