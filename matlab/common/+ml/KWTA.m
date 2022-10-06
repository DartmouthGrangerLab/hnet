% Eli Bowen 2/2022
% INPUTS
%   data - d x n (numeric)
%   k    - scalar or n x 1 (int-valued numeric)
% RETURNS
%   data - d x n (numeric) if input is on gpu or sparse, so is output
%   idx  - k x n (numeric index 1 --> d), elements are sorted largest to smallest
% see also ml.OneWTA
function data = KWTA(data, k)
    validateattributes(data, {'numeric'}, {}, 1);
    validateattributes(k, {'numeric'}, {'nonempty','integer'}, 2);
    
    if isscalar(k) && k == 1
        data = ml.OneWTA(data);
        return
    end
    
    do_flip = false;
    if isvector(data) && size(data, 2) > size(data, 1)
        data = data'; % should be d x 1
        do_flip = true;
    end
    [~,n] = size(data);

    if isscalar(k)
        [~,idx] = maxk(data, k, 1); % idx is k x n
        data(:) = 0;
        for i = 1 : n
            data(idx(:,i),i) = 1;
        end
    else
        [~,idx] = maxk(data, max(k), 1); % idx is k x n, sorted from max downwards
        data(:) = 0;
        for i = 1 : n
            data(idx(1:k(i),i),i) = 1;
        end
    end

    if do_flip
        data = data'; % flip back
    end
end