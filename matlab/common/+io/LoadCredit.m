% Eli Bowen 3/2022
% INPUTS
%   name - (char) name of dataset
%   path - OPTIONAL (char) directory in which to find the dataset file(s)
% RETURNS
%   dataset - (struct) with fields:
%       .t      - n x d (table) dataset
%       .t_bin  - n x X (table) dataset with categorical variables binarized (continuous variables unchanged)
%       .uniq_* - 1 x Y (cell array of chars)
function dataset = LoadCredit(name, path)
    validateattributes(name, {'char'}, {'nonempty'}, 1);

    if ~exist('path', 'var') || isempty(path)
        path = fullfile(ComputerProfile.DatasetDir(), 'credit', name);
    end

    dataset = struct();
    if strcmp(name, 'uci_credit_screening')
        dataset.t = readtable(fullfile(path, 'crx.csv'));
        dataset.t.Properties.VariableNames = {'a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','dv'};
        
        % convert binary vars to logical datatype
        dataset.t.a9  = strcmp(dataset.t.a9, 't');
        dataset.t.a10 = strcmp(dataset.t.a10, 't');
        dataset.t.a12 = strcmp(dataset.t.a12, 't');
        dataset.t.dv  = strcmp(dataset.t.dv, '+'); % make logical for simplicity
        
        % get unique values for categorical vars
        dataset.uniq_a1  = {'a','b'};
        dataset.uniq_a4  = {'u','y','l','t'};
        dataset.uniq_a5  = {'g','p','gg'};
        dataset.uniq_a6  = {'c','d','cc','i','j','k','m','r','q','w','x','e','aa','ff'};
        dataset.uniq_a7  = {'v','h','bb','j','n','z','dd','ff','o'};
        dataset.uniq_a13 = {'g','p','s'};
        
        % binarize categorical vars
        dataset.t_bin = dataset.t;
        for i = 1 : numel(dataset.uniq_a1)
            dataset.t_bin.(['a1_',dataset.uniq_a1{i}]) = strcmp(dataset.t.a1, dataset.uniq_a1{i});
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a1');
        for i = 1 : numel(dataset.uniq_a4)
            dataset.t_bin.(['a4_',dataset.uniq_a4{i}]) = strcmp(dataset.t.a4, dataset.uniq_a4{i});
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a4');
        for i = 1 : numel(dataset.uniq_a5)
            dataset.t_bin.(['a5_',dataset.uniq_a5{i}]) = strcmp(dataset.t.a5, dataset.uniq_a5{i});
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a5');
        for i = 1 : numel(dataset.uniq_a6)
            dataset.t_bin.(['a6_',dataset.uniq_a6{i}]) = strcmp(dataset.t.a6, dataset.uniq_a6{i});
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a6');
        for i = 1 : numel(dataset.uniq_a7)
            dataset.t_bin.(['a7_',dataset.uniq_a7{i}]) = strcmp(dataset.t.a7, dataset.uniq_a7{i});
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a7');
        for i = 1 : numel(dataset.uniq_a13)
            dataset.t_bin.(['a13_',dataset.uniq_a13{i}]) = strcmp(dataset.t.a13, dataset.uniq_a13{i});
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a13');
    elseif strcmp(name, 'uci_statlog_australian_credit')
        dataset.t = readtable(fullfile(path, 'australian.csv'));
        dataset.t.Properties.VariableNames = {'a1_idx','a2','a3','a4_idx','a5_idx','a6_idx','a7','a8','a9','a10','a11','a12_idx','a13','a14','dv'};
        dataset.t.a1_idx = dataset.t.a1_idx + 1;
        dataset.t.a8  = logical(dataset.t.a8);
        dataset.t.a9  = logical(dataset.t.a9);
        dataset.t.a11 = logical(dataset.t.a11);
        dataset.t.dv  = logical(dataset.t.dv); % despite the description (says values should be 1,2), seems to be logical
        dataset.uniq_a1 = {'a','b'};
        dataset.uniq_a4 = {'p','g','gg'};
        dataset.uniq_a5 = {'ff','d','i','k','j','aa','m','c','w','e','q','r','cc','x'};
        dataset.uniq_a6 = {'ff','dd','j','bb','v','n','o','h','z'};
        dataset.uniq_a12 = {'s','g','p'};
        dataset.t_bin = dataset.t;
        dataset.t_bin.a1 = logical(dataset.t.a1_idx - 1); % it's 2 classes
        dataset.t_bin = removevars(dataset.t_bin, 'a1_idx');
        dataset.t_bin.a4_p  = (dataset.t.a4_idx == 1);
        dataset.t_bin.a4_g  = (dataset.t.a4_idx == 2);
        dataset.t_bin.a4_gg = (dataset.t.a4_idx == 3);
        dataset.t_bin = removevars(dataset.t_bin, 'a4_idx');
        for i = 1 : numel(dataset.uniq_a5)
            dataset.t_bin.(['a5_',dataset.uniq_a5{i}]) = (dataset.t.a5_idx == i);
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a5_idx');
        for i = 1 : numel(dataset.uniq_a6)
            dataset.t_bin.(['a6_',dataset.uniq_a6{i}]) = (dataset.t.a6_idx == i);
        end
        dataset.t_bin = removevars(dataset.t_bin, 'a6_idx');
        dataset.t_bin.a12_s = (dataset.t.a12_idx == StringFind(dataset.uniq_a12, 's', true));
        dataset.t_bin.a12_g = (dataset.t.a12_idx == StringFind(dataset.uniq_a12, 'g', true));
        dataset.t_bin.a12_p = (dataset.t.a12_idx == StringFind(dataset.uniq_a12, 'p', true));
        dataset.t_bin = removevars(dataset.t_bin, 'a12_idx');
    elseif strcmp(name, 'uci_statlog_german_credit')
        dataset.t = readtable(fullfile(path, 'german.txt'));
        dataset.t.Properties.VariableNames = {'a1_idx','a2_duration','a3_idx','a4_idx','a5_creditscore','a6_idx','a7_idx','a8_percent','a9_idx','a10_idx','a11_presentresidencesince','a12_idx','a13_age','a14_idx','a15_idx','a16_ncredits','a17_idx','a18_ndependents','a19_hasphone','a20_isforeign','dv'};
        
        % convert binary vars to logical datatype
        dataset.t.a19_hasphone = strcmp(dataset.t.a19_hasphone, 'A192');
        dataset.t.a20_isforeign = strcmp(dataset.t.a20_isforeign, 'A201');
        dataset.t.dv = logical(dataset.t.dv - 1); % make logical for simplicity
        
        % add a lookup table for definitions of category names
        dataset.category_info = struct();
        dataset.category_info.a1_A11 = 'salary for at least 1 year = under zero dm'; % a1 status of existing checking account
        dataset.category_info.a1_A12 = 'salary for at least 1 year = under 200 dm'; % a1 status of existing checking account
        dataset.category_info.a1_A13 = 'salary for at least 1 year = over 200 dm'; % a1 status of existing checking account
        dataset.category_info.a1_A14 = 'no checking acct'; % a1 status of existing checking account
        
        dataset.category_info.a3_A30 = 'no credits taken / all credits paid back duly'; % a3 credit history
        dataset.category_info.a3_A31 = 'all credits at this bank paid back duly'; % a3 credit history
        dataset.category_info.a3_A32 = 'existing credits paid back duly till now'; % a3 credit history
        dataset.category_info.a3_A33 = 'delay in paying off in the past'; % a3 credit history
        dataset.category_info.a3_A34 = 'critical account / other credits existing (not at this bank)'; % a3 credit history

        dataset.category_info.a4_A40 = 'purpose = car (new)'; % a4 purpose
        dataset.category_info.a4_A41 = 'purpose = car (used)'; % a4 purpose
        dataset.category_info.a4_A42 = 'purpose = furniture / equipment'; % a4 purpose
        dataset.category_info.a4_A43 = 'purpose = radio / television'; % a4 purpose
        dataset.category_info.a4_A44 = 'purpose = domestic appliances'; % a4 purpose
        dataset.category_info.a4_A45 = 'purpose = repairs'; % a4 purpose
        dataset.category_info.a4_A46 = 'purpose = education'; % a4 purpose
        dataset.category_info.a4_A47 = 'purpose = (vacation - does not exist?)'; % a4 purpose
        dataset.category_info.a4_A48 = 'purpose = retraining'; % a4 purpose
        dataset.category_info.a4_A49 = 'purpose = business'; % a4 purpose
        dataset.category_info.a4_A410 = 'purpose = other'; % a4 purpose

        dataset.category_info.a6_A61 = 'savings and bonds = < 100 dm'; % a6 savings account / bonds
        dataset.category_info.a6_A62 = 'savings and bonds = 100 to 500 dm'; % a6 savings account / bonds
        dataset.category_info.a6_A63 = 'savings and bonds = 500 to 1000 dm'; % a6 savings account / bonds
        dataset.category_info.a6_A64 = 'savings and bonds = over 1000 dm'; % a6 savings account / bonds
        dataset.category_info.a6_A65 = 'savings and bonds = unknown / none'; % a6 savings account / bonds

        dataset.category_info.a7_A71 = 'unemployed'; % a7 present employment since
        dataset.category_info.a7_A72 = 'present job held < 1 year'; % a7 present employment since
        dataset.category_info.a7_A73 = 'present job held 1 to 4 years'; % a7 present employment since
        dataset.category_info.a7_A74 = 'present job held 4 to 7 years'; % a7 present employment since
        dataset.category_info.a7_A75 = 'present job held over 7 years'; % a7 present employment since

        dataset.category_info.a9_A91 = 'male, divorced / separated'; % a9 personal status and sex
        dataset.category_info.a9_A92 = 'female, divorced / separated / married'; % a9 personal status and sex
        dataset.category_info.a9_A93 = 'male, single'; % a9 personal status and sex
        dataset.category_info.a9_A94 = 'male, married / widowed'; % a9 personal status and sex
        dataset.category_info.a9_A95 = 'female, single'; % a9 personal status and sex

        dataset.category_info.a10_A101 = 'other debtors / guarantors = none'; % a10 other debtors / guarantors
        dataset.category_info.a10_A102 = 'other debtors / guarantors = co-applicant'; % a10 other debtors / guarantors
        dataset.category_info.a10_A103 = 'other debtors / guarantors = guarantor'; % a10 other debtors / guarantors

        dataset.category_info.a12_A121 = 'property = real estate'; % a12 property
        dataset.category_info.a12_A122 = 'property = if not A121 : building society savings agreement / life insurance'; % a12 property
        dataset.category_info.a12_A123 = 'property = if not A121/A122 : car or other, not in attribute 6'; % a12 property
        dataset.category_info.a12_A124 = 'property = unknown / none'; % a12 property

        dataset.category_info.a14_A141 = 'other installment plans = bank'; % a14 other installment plans
        dataset.category_info.a14_A142 = 'other installment plans = stores'; % a14 other installment plans
        dataset.category_info.a14_A143 = 'other installment plans = none'; % a14 other installment plans

        dataset.category_info.a15_A151 = 'housing = rent'; % a15 housing
        dataset.category_info.a15_A152 = 'housing = own'; % a15 housing
        dataset.category_info.a15_A153 = 'housing = for free'; % a15 housing

        dataset.category_info.a17_A171 = 'job = unemployed / unskilled - non-resident'; % a17 job
        dataset.category_info.a17_A172 = 'job = unskilled - resident'; % a17 job
        dataset.category_info.a17_A173 = 'job = skilled employee / official'; % a17 job
        dataset.category_info.a17_A174 = 'job = management / self-employed / highly qualified employee / officer'; % a17 job
        
        % get unique values for categorical vars
        for i = [1,3,4,6,7,9,10,12,14,15,17]
            dataset.(['uniq_a',num2str(i)]) = unique(table2cell(dataset.t(:,i)));
%             [dataset.(['uniq_a',num2str(i)]),~,dataset.t.(['a',num2str(i),'_idx'])]  = unique(dataset.t(:,i));
        end
        
        % binarize categorical vars
        dataset.t_bin = dataset.t;
        for i = [1,3,4,6,7,9,10,12,14,15,17]
            varName = ['a',num2str(i)];
            for j = 1 : numel(dataset.(['uniq_',varName]))
                dataset.t_bin.([varName,'_',dataset.(['uniq_',varName]){j}]) = strcmp(dataset.t.([varName,'_idx']), dataset.(['uniq_',varName]){j});
            end
            dataset.t_bin = removevars(dataset.t_bin, [varName,'_idx']);
        end
        % continuous vars: a2_duration, a5_creditscore, a8_percent (uniques=1,2,3,4), a11_presentresidencesince (uniques=1,2,3,4), a13_age, 16_ncredits (uniques={1,2,3,4}), a18_ndependents (uniques=1,2)
    elseif strcmp(name, 'kaggle_icl_loan_default_prediction')
        error('TODO');
    else
        error('unexpected name');
    end
end