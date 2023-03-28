% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
% print edges to output
function [] = PrintEdgeRelations(path, diaryname, edgeStates, didx, nodeName, bank, tier1Bank)
    arguments
        path(1,1) string, diaryname(1,1) string, edgeStates(:,1) EDG, didx(:,2), nodeName cell, bank(1,1) string, tier1Bank(1,1) string
    end
    
    str = "";
    
    for ii = 1 : 16
        if any(edgeStates == ii)
            idx = find(edgeStates == ii);
            
            current = didx(idx,:);
            if EDG(ii) == EDG.NCONV
                [~,sortIdx] = sort(current(:,2)); % sort by the second node
                current = current(sortIdx,:);
            end
            
            if EDG(ii) == EDG.NCONV || EDG(ii) == EDG.NIMPL
                relationStr = " AND NOT ";
            else
                relationStr = " " + char(EDG(ii)) + " ";
            end
            
            for i = 1 : size(current, 1)
                if EDG(ii) == EDG.NCONV
                    if i > 1 && strcmpi(nodeName(current(i,2)), nodeName(current(i-1,2))) && edgeStates(idx(i)) == edgeStates(idx(i-1))
                        str = str + " " + relationStr + nodeName(current(i,1)) + newline();
                    else
                        str = str + nodeName(current(i,2)) + relationStr + nodeName(current(i,1)) + newline();
                    end
                else
                    if i > 1 && strcmpi(nodeName(current(i,1)), nodeName(current(i-1,1))) && edgeStates(idx(i)) == edgeStates(idx(i-1))
                        str = str + " " + relationStr + nodeName(current(i,2)) + newline();
                    else
                        str = str + nodeName(current(i,1)) + relationStr + nodeName(current(i,2)) + newline();
                    end
                end
            end

            if bank == "meta"
                parts = unique(didx(idx,:));
                for i = 1 : numel(parts)
                    str = str + " " + newline() + "CURRENT PART: " + int2str(parts(i));
                    str = str + newline() + fileread(fullfile(path, "relations", tier1Bank + "_cmp" + int2str(parts(i)) + ".txt"));
                end
            end
            
            str = str + newline();
        end
    end
    
    if ~isfolder(fullfile(path, "relations"))
        mkdir(path);
        mkdir(fullfile(path, "relations"));
    end
    fid = fopen(fullfile(path, "relations", diaryname), 'w');
    fprintf(fid, "%s", str);
    fclose(fid);
end
