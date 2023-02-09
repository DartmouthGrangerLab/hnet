% Eli Bowen 10/26/2022
% INPUTS
%   subset - (char) 'train' | 'test' | 'val'
%   imgScaleFactor - scalar (numeric) 1 = original quality
%   path - OPTIONAL (char) directory in which to find the dataset file(s)
% RETURNS
%   dataset - (struct) with fields:
%       .info                    - (struct) boring dataset metadata;
%       .n                       - scalar (int-valued numeric) number of images in this subset
%       .image_idx               - n x 1 (numeric)
%       .objects                 - n x 1 (cell)
%       .relationships_right{i}  - n x 1 (cell)
%       .relationships_behind{i} - n x 1 (cell)
%       .relationships_front{i}  - n x 1 (cell)
%       .relationships_left{i}   - n x 1 (cell)
%       .image_filename          - n x 1 (cell)
%       .directions_right        - n x 3 (numeric)
%       .directions_behind       - n x 3 (numeric)
%       .directions_above        - n x 3 (numeric)
%       .directions_below        - n x 3 (numeric)
%       .directions_left         - n x 3 (numeric)
%       .directions_front        - n x 3 (numeric)
%       .questions               - (struct)
%       .img
% USAGE
%   dataset = CachedCompute(@io.LoadCLEVR, 'train', 0.125)
function dataset = LoadCLEVR(subset, imgScaleFactor, path)
    if ~exist('path', 'var') || isempty(path)
        path = fullfile(ComputerProfile.DatasetDir(), 'img_clevr', 'CLEVR_v1.0');
    end
    
    dataset = struct();

    % struct with fields:
    %   .info
    %       .split
    %       .license
    %       .version
    %       .date
    %   .scenes - 70K x 1 struct array, each entry containing:
    %       .image_index    - scalar integer starting from 0
    %       .objects        - variable-length struct array, always with fields:
    %           .color        - (char) e.g. 'purple'
    %           .size         - (char) e.g. 'small'
    %           .rotation     - scalar (numeric) looks like degrees
    %           .shape        - (char) e.g. 'sphere'
    %           .x3d_coords   - 3 x 1 (numeric)
    %           .material     - (char) e.g. 'metal'
    %           .pixel_coords - 3 x 1 (numeric) the first 2 look like pixels, no idea what the third dimension is...
    %       .relationships  - struct
    %           .right  - ? x 1 (cell) each entry a different length numeric array
    %           .behind - ? x 1 (cell) each entry a different length numeric array
    %           .front  - ? x 1 (cell) each entry a different length numeric array
    %           .left   - ? x 1 (cell) each entry a different length numeric array
    %       .image_filename - (char) not including path
    %       .split          - (char) e.g. 'train'
    %       .directions     - struct
    %           .right  - 3 x 1 (numeric)
    %           .behind - 3 x 1 (numeric)
    %           .above  - 3 x 1 (numeric)
    %           .below  - 3 x 1 (numeric)
    %           .left   - 3 x 1 (numeric)
    %           .front  - 3 x 1 (numeric)
    temp = jsondecode(io.UnzipText(fullfile(path, 'scenes', ['CLEVR_',subset,'_scenes.json.zip'])));
    dataset.info = temp.info;
    dataset.n = numel(temp.scenes);
    dataset.image_idx            = zeros(dataset.n, 1);
    dataset.objects              = cell(dataset.n, 1);
    dataset.relationships_right  = cell(dataset.n, 1);
    dataset.relationships_behind = cell(dataset.n, 1);
    dataset.relationships_front  = cell(dataset.n, 1);
    dataset.relationships_left   = cell(dataset.n, 1);
    dataset.image_filename       = cell(dataset.n, 1);
    dataset.directions_right     = zeros(dataset.n, 3);
    dataset.directions_behind    = zeros(dataset.n, 3);
    dataset.directions_above     = zeros(dataset.n, 3);
    dataset.directions_below     = zeros(dataset.n, 3);
    dataset.directions_left      = zeros(dataset.n, 3);
    dataset.directions_front     = zeros(dataset.n, 3);
    for i = 1 : dataset.n
        assert(strcmp(temp.scenes(i).split, subset));
        dataset.image_idx(i)      = temp.scenes(i).image_index + 1; % convert from 0-based to 1-based indexing
        dataset.objects{i}        = temp.scenes(i).objects;
        dataset.relationships_right{i}  = temp.scenes(i).relationships.right;
        dataset.relationships_behind{i} = temp.scenes(i).relationships.behind;
        dataset.relationships_front{i}  = temp.scenes(i).relationships.front;
        dataset.relationships_left{i}   = temp.scenes(i).relationships.left;
        dataset.image_filename{i}       = temp.scenes(i).image_filename;
        dataset.directions_right(i,:)   = temp.scenes(i).directions.right;
        dataset.directions_behind(i,:)  = temp.scenes(i).directions.behind;
        dataset.directions_above(i,:)   = temp.scenes(i).directions.above;
        dataset.directions_below(i,:)   = temp.scenes(i).directions.below;
        dataset.directions_left(i,:)    = temp.scenes(i).directions.left;
        dataset.directions_front(i,:)   = temp.scenes(i).directions.front;
    end
    
    % struct with fields:
    %   .info
    %       .split
    %       .license
    %       .version
    %       .date
    %   .questions - 70K x 1 struct array with fields:
    %       .image_index           - scalar integer starting from 0
    %       .program               - variable-length struct array, always with fields:
    %           .inputs
    %           .function
    %           .value_inputs
    %       .question_index        - scalar integer starting from 0
    %       .image_filename        - (char) not including path
    %       .question_family_index - scalar integer
    %       .split                 - (char) e.g. 'train'
    %       .answer                - (char)
    %       .question              - (char)
    temp = jsondecode(io.UnzipText(fullfile(path, 'questions', ['CLEVR_',subset,'_questions.json.zip'])));
    dataset.questions = struct();
    dataset.questions.image_idx           = zeros(dataset.n, 1);
    dataset.questions.question_idx        = zeros(dataset.n, 1);
    dataset.questions.question_family_idx = zeros(dataset.n, 1);
    dataset.questions.question            = cell(dataset.n, 1);
    dataset.questions.answer              = cell(dataset.n, 1);
    for i = 1 : dataset.n
        assert(strcmp(temp.questions(i).image_filename, dataset.image_filename(dataset.image_idx == temp.questions(i).image_index+1)));
        assert(strcmp(temp.questions(i).split, subset));
        %TODO: parse .program if needed
        dataset.questions.image_idx(i)           = temp.questions(i).image_index + 1; % convert from 0-based to 1-based indexing
        dataset.questions.question_idx(i)        = temp.questions(i).question_index + 1; % convert from 0-based to 1-based indexing
        dataset.questions.question_family_idx(i) = temp.questions(i).question_family_index + 1; % convert from 0-based to 1-based indexing
        dataset.questions.question{i}            = temp.questions(i).question;
        dataset.questions.answer{i}              = temp.questions(i).answer;
    end

    dataset.img = zeros(320*imgScaleFactor, 480*imgScaleFactor, 3, dataset.n, 'uint8'); % 480x320 px color

    unzipDir = fullfile(ComputerProfile.CacheDir(), ['loadclevr_',GetMD5(now, 'array', 'hex'),'_',subset]);
    assert(~isfolder(unzipDir));
    unzip(fullfile(path, 'images', [subset,'.zip']), unzipDir);
    for i = 1 : dataset.n
        if imgScaleFactor == 1
            dataset.img(:,:,:,i) = imread(fullfile(unzipDir, subset, dataset.image_filename{i}));
        else
            dataset.img(:,:,:,i) = imresize(imread(fullfile(unzipDir, subset, dataset.image_filename{i})), imgScaleFactor);
        end
    end
    delete(fullfile(unzipDir, subset, '*'));
    rmdir(fullfile(unzipDir, '*'));
    rmdir(unzipDir);

    for i = 1 : dataset.n
        for j = 1 : numel(dataset.objects{i})
            dataset.objects{i}(j).pixel_coords(1:2) = dataset.objects{i}(j).pixel_coords(1:2) .* imgScaleFactor;
        end
    end
end