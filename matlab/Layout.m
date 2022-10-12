% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
% INPUTS
%   name - (char) name of layout to load
% RETURNS
%   layout - (cell array of structs)
function layout = Layout(name)
    arguments
        name (1,:) char
    end
    
    %% load
%     layout = jsondecode(fileread(['layout_',name,'.json'])); % json file must be on the matlab path
    layout = {};
    if strcmp(name, 'basicimg')
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout{2} = struct(connec='sense-->connectedpart,connectedpart-->out');
    elseif strcmp(name, 'basiccred')
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout{2} = struct(connec='sense-->tier1,tier1-->out');
    elseif strcmp(name, 'basiccredand')
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=EDG.AND, encode_spec='energy');
        layout{2} = struct(connec='sense-->tier1,tier1-->out');
    elseif strcmp(name, 'groupedimg')
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout{2} = struct(name='group',         graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='max');
        layout{3} = struct(connec='sense-->connectedpart,connectedpart-->group,group-->out');
    elseif strcmp(name, 'groupedcred')
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout{2} = struct(name='group', graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max');
        layout{3} = struct(connec='sense-->tier1,tier1-->group,group-->out');
    elseif strcmp(name, 'groupedwta20img')
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout{2} = struct(name='group',         graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='max-->wta.20');
        layout{3} = struct(connec='sense-->connectedpart,connectedpart-->group,group-->out');
    elseif strcmp(name, 'groupedwta20cred')
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout{2} = struct(name='group', graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max-->wta.20');
        layout{3} = struct(connec='sense-->tier1,tier1-->group,group-->out');
    elseif strcmp(name, 'groupedabsimg')
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout{2} = struct(name='group',         graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='maxabs');
        layout{3} = struct(connec='sense-->connectedpart,connectedpart-->group,group-->out');
    elseif strcmp(name, 'groupedabscred')
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout{2} = struct(name='group', graph_type=GRF.SELF, edge_type_filter=[], encode_spec='maxabs');
        layout{3} = struct(connec='sense-->tier1,tier1-->group,group-->out');
    elseif strcmp(name, 'groupedabswta20img')
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout{2} = struct(name='group',         graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='maxabs-->wta.20');
        layout{3} = struct(connec='sense-->connectedpart,connectedpart-->group,group-->out');
    elseif strcmp(name, 'groupedabswta20cred')
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy');
        layout{2} = struct(name='group', graph_type=GRF.SELF, edge_type_filter=[], encode_spec='maxabs-->wta.20');
        layout{3} = struct(connec='sense-->tier1,tier1-->group,group-->out');
    elseif strcmp(name, 'grouptransl2')
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energytransl.2');
        layout{2} = struct(name='group',         graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='max');
        layout{3} = struct(connec='sense-->connectedpart,connectedpart-->group,group-->out');
    elseif strcmp(name, 'metaimg') % meta hierarchy for image datasets
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy-->wta.20');
        layout{2} = struct(name='meta',          graph_type=GRF.FULL,      edge_type_filter=[],                    encode_spec='energy');
        layout{3} = struct(connec='sense-->connectedpart,connectedpart-->meta,meta-->out');
    elseif strcmp(name, 'metacred') % meta hierarchy for credit datasets
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND], encode_spec='energy-->wta.20');
        layout{2} = struct(name='meta',  graph_type=GRF.FULL, edge_type_filter=[EDG.NCONV,EDG.NIMPL,EDG.AND],      encode_spec='energy');
        layout{3} = struct(connec='sense-->tier1,tier1-->meta,meta-->out');
    elseif strcmp(name, 'metacredand') % meta hierarchy for credit datasets
        layout{1} = struct(name='tier1', graph_type=GRF.FULL, edge_type_filter=EDG.AND, encode_spec='energy-->wta.20');
        layout{2} = struct(name='meta',  graph_type=GRF.FULL, edge_type_filter=[],      encode_spec='energy');
        layout{3} = struct(connec='sense-->tier1,tier1-->meta,meta-->out');
    elseif strcmp(name, 'metagrpimg') % meta hierarchy for image datasets
        layout{1} = struct(name='connectedpart', graph_type=GRF.GRID2DSQR, edge_type_filter=[EDG.NCONV,EDG.NIMPL], encode_spec='energy');
        layout{2} = struct(name='group',         graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='max');
        layout{3} = struct(name='meta',          graph_type=GRF.NULL, edge_type_filter=[], encode_spec='energy');
        layout{4} = struct(name='metagroup',     graph_type=GRF.SELF,      edge_type_filter=[], encode_spec='max');
        layout{5} = struct(connec='sense-->connectedpart,connectedpart-->group,group-->meta,meta-->metagroup,metagroup-->out');
    elseif strcmp(name, 'metagrpcred') % meta hierarchy for credit datasets
        layout{1} = struct(name='tier1',     graph_type=GRF.FULL, edge_type_filter=[], encode_spec='energy');
        layout{2} = struct(name='group',     graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max');
        layout{3} = struct(name='meta',      graph_type=GRF.FULL, edge_type_filter=[], encode_spec='energy');
        layout{4} = struct(name='metagroup', graph_type=GRF.SELF, edge_type_filter=[], encode_spec='max');
        layout{5} = struct(connec='sense-->tier1,tier1-->group,group-->meta,meta-->metagroup,metagroup-->out');
    else
        error('unexpected name');
    end
    
    %% parse fields
    if isfield(layout{1}, 'comment')
        layout = layout(2:end); % first entry is a description
    end
    for id = 1 : numel(layout)
        if isfield(layout{id}, 'graph_type')
            layout{id}.graph_type = GRF(layout{id}.graph_type); % convert to enum
        end
        if isfield(layout{id}, 'edge_type_filter')
            layout{id}.edge_type_filter = EDG(layout{id}.edge_type_filter); % convert to enum
        end
    end
    
    %% validate
    % layout{i} - struct
    %   .name             - (char) name of component bank
    %   .graph_type       - scalar (GRF enum) type of graph used by this component bank
    %   .edge_type_filter - vector (EDG enum) empty = no filtering
    %   .encode_spec      - (char) type of encoding employed by this component bank; see Encode() for options
    % - or -
    %   .connec - (char) connectivity
    for id = 1 : numel(layout)
        validateattributes(layout{id}, 'struct', {'nonempty','scalar'});
        
        if isfield(layout{id}, 'name')
            validateattributes(layout{id}.name,             'char', {'nonempty'});
            validateattributes(layout{id}.graph_type,       'GRF',  {'nonempty','scalar','positive','integer'});
            validateattributes(layout{id}.edge_type_filter, 'EDG',  {});
            validateattributes(layout{id}.encode_spec,      'char', {'nonempty'});
            assert(~strcmpi(layout{id}.name, 'sense') && ~strcmpi(layout{id}.name, 'label')); % reserved
        else
            validateattributes(layout{id}.connec, 'char', {'nonempty'});
        end
    end
end