% Eli Bowen 5/2022
% INPUTS:
%   path   - (char) table file directory
%   file   - (char) table file name
%   keyVar - (char) name of variable to use as struct
%   s      - struct
function [] = InjectRowIntoTableFile(path, file, keyVar, s)
    validateattributes(path, {'char'}, {'nonempty'}, 1);
    validateattributes(file, {'char'}, {'nonempty'}, 2);
    validateattributes(keyVar, {'char'}, {'nonempty'}, 3);
    validateattributes(s, {'struct'}, {}, 4);

    key = s.(keyVar);

    varNames = fieldnames(s);
    data = cell(1, numel(varNames));
    for i = 1 : numel(varNames)
        data{i} = s.(varNames{i});
    end

    % load or initialize the table
    if isfile(fullfile(path, file))
        t = readtable(fullfile(path, file));
        for i = 1 : numel(varNames)
            assert(strcmp(varNames{i}, t.Properties.VariableNames{i}));
        end
    else
        warning('off', 'MATLAB:table:PreallocateCharWarning'); % shh
        t = table('Size', [0,numel(data)], 'VariableTypes', cellfun(@class, data, 'UniformOutput', false), 'VariableNames', varNames);
    end

    % find the right row for this data
    row = [];
    if ~isempty(t)
        if ischar(key)
            row = StringFind(table2cell(t(:,1)), key, true); % overwrite if possible
        elseif isnumeric(key) && isscalar(key)
            row = find(table2array(t(:,1)) == key);
        else
            error('unexpected datatype for keyVar');
        end
    end
    if isempty(row)
        row = size(t, 1) + 1; % append to the end
    end

    t(row,:) = data;
    writetable(t, fullfile(path, file));
end