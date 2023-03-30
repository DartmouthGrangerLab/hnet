% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% INPUTS
%   name - (char) name of layout to load
% RETURNS
%   layout - (cell array of structs)
function layout = Layout(name)
    arguments
        name(1,:) char
    end
    
    %% load
%     layout = jsondecode(fileread(['layout_',name,'.json'])); % json file must be on the matlab path

    layout = struct();
    if name == "basicimg"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.connec = 'sense-->connectedpart,connectedpart-->out';
    elseif name == "basiccred"
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout.connec = 'sense-->tier1,tier1-->out';
    elseif name == "basiccredand"
        layout = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=EDG.AND, encode_spec='energy');
        layout.connec = 'sense-->tier1,tier1-->out';
    elseif name == "groupedimg"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.group         = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='max');
        layout.connec = 'sense-->connectedpart,connectedpart-->group,group-->out';
    elseif name == "groupedcred"
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout.group = struct(graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max');
        layout.connec = 'sense-->tier1,tier1-->group,group-->out';
    elseif name == "groupedwta20img"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.group         = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='max-->wta.20');
        layout.connec = 'sense-->connectedpart,connectedpart-->group,group-->out';
    elseif name == "groupedwta20cred"
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout.group = struct(graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max-->wta.20');
        layout.connec = 'sense-->tier1,tier1-->group,group-->out';
    elseif name == "groupedabsimg"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.group         = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='maxabs');
        layout.connec = 'sense-->connectedpart,connectedpart-->group,group-->out';
    elseif name == "groupedabscred"
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout.group = struct(graph_type=GRF.SELF, edge_type_filter=[], encode_spec='maxabs');
        layout.connec = 'sense-->tier1,tier1-->group,group-->out';
    elseif name == "groupedabswta20img"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.group         = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='maxabs-->wta.20');
        layout.connec = 'sense-->connectedpart,connectedpart-->group,group-->out';
    elseif name == "groupedabswta20cred"
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout.group = struct(graph_type=GRF.SELF, edge_type_filter=[], encode_spec='maxabs-->wta.20');
        layout.connec = 'sense-->tier1,tier1-->group,group-->out';
    elseif name == "grouptransl2"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energytransl.2');
        layout.group         = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='max');
        layout.connec = 'sense-->connectedpart,connectedpart-->group,group-->out';
    elseif name == "metaimg" % meta hierarchy for image datasets
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy-->wta.20');
        layout.meta          = struct(graph_type=GRF.FULL,   edge_type_filter=[],                    encode_spec='energy');
        layout.connec = 'sense-->connectedpart,connectedpart-->meta,meta-->out';
    elseif name == "metacred" % meta hierarchy for credit datasets
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy-->wta.20');
        layout.meta  = struct(graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND],      encode_spec='energy');
        layout.connec = 'sense-->tier1,tier1-->meta,meta-->out';
    elseif name == "metacredand" % meta hierarchy for credit datasets
        layout.tier1 = struct(graph_type=GRF.FULL, edge_type_filter=EDG.AND, encode_spec='energy-->wta.20');
        layout.meta  = struct(graph_type=GRF.FULL, edge_type_filter=[],      encode_spec='energy');
        layout.connec = 'sense-->tier1,tier1-->meta,meta-->out';
    elseif name == "metagrpimg" % meta hierarchy for image datasets
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.group         = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='max');
        layout.meta          = struct(graph_type=GRF.NULL, edge_type_filter=[], encode_spec='energy');
        layout.metagroup     = struct(graph_type=GRF.SELF,   edge_type_filter=[], encode_spec='max');
        layout.connec = 'sense-->connectedpart,connectedpart-->group,group-->meta,meta-->metagroup,metagroup-->out';
    elseif name == "metagrpcred" % meta hierarchy for credit datasets
        layout.tier1     = struct(graph_type=GRF.FULL, edge_type_filter=[], encode_spec='energy');
        layout.group     = struct(graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max');
        layout.meta      = struct(graph_type=GRF.FULL, edge_type_filter=[], encode_spec='energy');
        layout.metagroup = struct(graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max');
        layout.connec = 'sense-->tier1,tier1-->group,group-->meta,meta-->metagroup,metagroup-->out';
    elseif name == "clevr"
        layout.connectedpart = struct(graph_type=GRF.GRID2D, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout.meta          = struct(graph_type=GRF.FULL,   edge_type_filter=[EDG.AND],             encode_spec='energy');
        layout.connec = 'sense-->connectedpart,connectedpart-->meta,meta-->out';
    else
        error("unexpected name");
    end
    
    %% parse fields
    if isfield(layout, "comment")
        layout = rmfield(layout, "comment"); % remove a comment if any
    end
    fn = fieldnames(layout);
    for i = 1 : numel(fn)
        if isfield(layout.(fn{i}), "graph_type")
            layout.(fn{i}).graph_type = GRF(layout.(fn{i}).graph_type); % convert to enum
        end
        if isfield(layout.(fn{i}), "edge_type_filter")
            layout.(fn{i}).edge_type_filter = EDG(layout.(fn{i}).edge_type_filter); % convert to enum
        end
    end
    
    %% validate
    % layout.field is a string or a struct with fields:
    %   .graph_type       - scalar (GRF enum) type of graph used by this component bank
    %   .edge_type_filter - vector (EDG enum) empty = no filtering
    %   .encode_spec      - (char) type of encoding employed by this component bank; see Encode() for options
    validateattributes(layout, {'struct'}, {'nonempty','scalar'});
    fn = fieldnames(layout);
    for id = 1 : numel(fn)
        assert(~strcmpi(fn{i}, "sense") && ~strcmpi(fn{i}, "label")); % reserved
        validateattributes(layout.(fn{i}), {'struct','char'}, {'nonempty'});
        if strcmp(fn{i}, "connec")
            validateattributes(layout.(fn{i}), 'char', {'nonempty'});
        else
            validateattributes(layout.(fn{i}).graph_type,       'GRF',  {'nonempty','scalar','positive','integer'});
            validateattributes(layout.(fn{i}).edge_type_filter, 'EDG',  {});
            validateattributes(layout.(fn{i}).encode_spec,      'char', {'nonempty'});
        end
    end
end