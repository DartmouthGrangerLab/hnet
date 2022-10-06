% Eli Bowen 5/22/2020
% the cells will be concatenated into a single vector
% this works even if they are of unequal dimensionality
% INPUTS
%   cellData - (cell)
%   sizeOfVal - OPTIONAL (for performance)
% RETURNS
%   val
function val = CellCat2Vec (cellData, sizeOfVal)
    validateattributes(cellData, {'cell'}, {'nonempty'}, 1);
    
%     if vectorize
    if ~exist('sizeOfVal', 'var') || isempty(sizeOfVal)
        sizeOfVal = sum(cellfun(@numel, cellData(:)));
    end

    val = zeros(1, sizeOfVal, 'like', cellData{1});
    count = 1;
    for i = 1:numel(cellData)
        val(count:count+numel(cellData{i})-1) = cellData{i}(:)';
        count = count + numel(cellData{i});
    end
    %instead of below, just call val = cat(dim, cellData{:});
%     else
%         if dim <= ndims(cellData{1})
%             val = cat(dim, cellData{:}); % just do what the people want
%         else
%             sizeOfVal = size(cellData{1});
%             if numel(sizeOfVal) == 2 && sizeOfVal(1) == 1
%                 sizeOfVal = sizeOfVal(2);
%             elseif numel(sizeOfVal) == 2 && sizeOfVal(2) == 1
%                 sizeOfVal = sizeOfVal(1);
%             end
%             val = zeros([sizeOfVal,numel(cellData)], 'like', cellData{1});
%             for i = 1:numel(cellData)
%                 if numel(sizeOfVal) == 1
%                     val(:,i) = cellData{i};
%                 elseif numel(sizeOfVal) == 2
%                     val(:,:,i) = cellData{i};
%                 elseif numel(sizeOfVal) == 3
%                     val(:,:,:,i) = cellData{i};
%                 elseif numel(sizeOfVal) == 4
%                     val(:,:,:,:,i) = cellData{i};
%                 elseif numel(sizeOfVal) == 5
%                     val(:,:,:,:,:,i) = cellData{i};
%                 else
%                     error('haven''t bothered to code this for ndims(cellData{i}) > 5');
%                 end
%             end
%         end
%     end
end