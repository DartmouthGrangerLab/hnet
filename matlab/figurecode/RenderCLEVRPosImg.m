% Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
% Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
% If you use this code, cite:
%   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
%   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
function RenderCLEVRPosImg(outDir, dat, model, imgidx)
    arguments
        outDir(1,1) string, dat(1,1) Dataset, model(1,1) Model, imgidx(1,1)
    end
    n_objects = dat.other_metadata.n_objects;
    scale = 4;
    fontsz = 10*scale;
    linewidth = scale;
    prototypeidx = 1; % use first image as the prototype

    img = imread(fullfile(Config.DATASET_DIR, "custom_clevr", "images", "customclevr_trnsimple_" + num2str(imgidx-1, "%06.f") + ".png"));
    img = imresize(img, scale);

    edge_endnode_src_idx = model.compbanks.tier1.g.edge_endnode_src_idx;
    edge_endnode_dst_idx = model.compbanks.tier1.g.edge_endnode_dst_idx;
    components = cell(model.compbanks.tier1.n_cmp, 1); % cell array of structs
    for i = 1 : model.compbanks.tier1.n_cmp
        idx = find(model.compbanks.tier1.edge_states(:,i));
        components{i}.obj1r   = zeros(numel(idx), 1);
        components{i}.obj1c   = zeros(numel(idx), 1);
        components{i}.obj1idx = zeros(numel(idx), 1);
        components{i}.obj2r   = zeros(numel(idx), 1);
        components{i}.obj2c   = zeros(numel(idx), 1);
        components{i}.obj2idx = zeros(numel(idx), 1);

        components{i}.obj1r(:) = model.compbanks.tier1.g.node_metadata.bucketr(edge_endnode_src_idx(idx));
        components{i}.obj1c(:) = model.compbanks.tier1.g.node_metadata.bucketc(edge_endnode_src_idx(idx));
        components{i}.obj1idx(:) = dat.label_metadata.nodeobjidxstate(edge_endnode_src_idx(idx),i);
        components{i}.obj2r(:) = model.compbanks.tier1.g.node_metadata.bucketr(edge_endnode_dst_idx(idx));
        components{i}.obj2c(:) = model.compbanks.tier1.g.node_metadata.bucketc(edge_endnode_dst_idx(idx));
        components{i}.obj2idx(:) = dat.label_metadata.nodeobjidxstate(edge_endnode_dst_idx(idx),i);
        assert(~any(components{i}.obj1idx == 0));
        assert(~any(components{i}.obj2idx == 0));
    end

    currcomp = components{imgidx};
    prototype = components{prototypeidx};

    if ~dat.label_metadata.is_face(imgidx)
        % find the position of each object in this image, and in the prototype
        imgobjr   = zeros(n_objects, 1);
        imgobjc   = zeros(n_objects, 1);
        protoobjr = zeros(n_objects, 1);
        protoobjc = zeros(n_objects, 1);
        for i = 1 : n_objects
            % taking mean, but really all of these values should be identical
            imgobjr(i)   = mean(cat(1, currcomp.obj1r(currcomp.obj1idx == i), currcomp.obj2r(currcomp.obj2idx == i)));
            imgobjc(i)   = mean(cat(1, currcomp.obj1c(currcomp.obj1idx == i), currcomp.obj2c(currcomp.obj2idx == i)));
            protoobjr(i) = mean(cat(1, prototype.obj1r(prototype.obj1idx == i), prototype.obj2r(prototype.obj2idx == i)));
            protoobjc(i) = mean(cat(1, prototype.obj1c(prototype.obj1idx == i), prototype.obj2c(prototype.obj2idx == i)));
        end

        for i = 1 : n_objects
            if i-1 == dat.label_metadata.randomized_obj_idx(imgidx)
                % compute vectors between this object and all other objects in the prototype
                allotherobjmask = ((1:n_objects) ~= i);
                diffr = protoobjr(i) - protoobjr(allotherobjmask);
                diffc = protoobjc(i) - protoobjc(allotherobjmask);

                % use those vectors, but starting from the current image's objects
                prototype.obj1r(prototype.obj1idx == i) = mean(diffr + imgobjr(allotherobjmask));
                prototype.obj1c(prototype.obj1idx == i) = mean(diffc + imgobjc(allotherobjmask));
                prototype.obj2r(prototype.obj2idx == i) = mean(diffr + imgobjr(allotherobjmask));
                prototype.obj2c(prototype.obj2idx == i) = mean(diffc + imgobjc(allotherobjmask));
            end
        end
    end
    
    randomized_obj_idx = dat.label_metadata.randomized_obj_idx(imgidx) + 1; % +1 for matlab's 1-based indexing
    if dat.label_metadata.is_face
        assert(randomized_obj_idx == 0);
    end
    img1 = RenderCLEVRPosImgBeforeAbduction(prototype, img, scale, fontsz, linewidth);
    img2 = RenderCLEVRPosImgAfterAbduction(prototype, img, scale, fontsz, linewidth, ~dat.label_metadata.eyes_same_color(imgidx), randomized_obj_idx);

    if dat.label_metadata.is_face
        intact = struct(twoeyes=true, eyesnose=true, mouth=true, face=true);
    else
        intact = struct(twoeyes=false, eyesnose=false, mouth=true, face=false);
    end
    intact_strings = {"¬ ",""};
    img2 = insertText(img2, [0,0],          intact_strings{intact.twoeyes+1} + "¢TwoEyes",  TextColor="yellow",  BoxColor="black", FontSize=fontsz);
    img2 = insertText(img2, [0,22*scale],   intact_strings{intact.twoeyes+1} + "¢EyesNose", TextColor="cyan",    BoxColor="black", FontSize=fontsz);
    img2 = insertText(img2, [0,2*22*scale], intact_strings{intact.twoeyes+1} + "¢Mouth",    TextColor="magenta", BoxColor="black", FontSize=fontsz);
    img2 = insertText(img2, [0,3*22*scale], intact_strings{intact.twoeyes+1} + "¢Face",     TextColor="green",   BoxColor="black", FontSize=fontsz);

    h = figure(Visible=false);
    if dat.label_metadata.is_face(imgidx) && dat.label_metadata.eyes_same_color(imgidx)
        fig.subplot(1, 2, 1, [0.05,0.05]);
        imshow(img1);
        title("before abduction");
        fig.subplot(1, 2, 2, [0.05,0.05]);
        imshow(img2);
        title("after abduction");
        fig.print(h, char(fullfile(outDir, "renderclevrposimages")), ['clevrposimg_',num2str(imgidx-1),'_match.png']);
    elseif dat.label_metadata.is_face(imgidx) && ~dat.label_metadata.eyes_same_color(imgidx)
        imshow(img2);
        title("inference, after abduction");
        fig.print(h, char(fullfile(outDir, "renderclevrposimages")), ['clevrposimg_',num2str(imgidx-1),'_eyesmismatch.png']);
    else % not dat.label_metadata.is_face
        imshow(img2);
        title("inference, after abduction");
        fig.print(h, char(fullfile(outDir, "renderclevrposimages")), ['clevrposimg_',num2str(imgidx-1),'_missingobject.png']);
    end
end


function img = RenderCLEVRPosImgBeforeAbduction(prototype, img, scale, fontsz, linewidth)
    n_edges = numel(prototype.obj1r);

    % tier 1, before abduction
    for i = 1 : n_edges % for each edge
        img = RenderLine(img, prototype.obj1r(i), prototype.obj1c(i), prototype.obj2r(i), prototype.obj2c(i), scale, linewidth, fontsz, "white", false);
    end

    % tier 2, before abduction
    linecentroidr = zeros(n_edges, 1);
    linecentroidc = zeros(n_edges, 1);
    for i = 1 : n_edges % for each edge (object pair)
        linecentroidr(i) = prototype.obj1r(i) + (prototype.obj2r(i) - prototype.obj1r(i)) / 2;
        linecentroidc(i) = prototype.obj1c(i) + (prototype.obj2c(i) - prototype.obj1c(i)) / 2;
    end
    for i = 1 : n_edges
        for j = i+1 : n_edges
            img = RenderLine(img, linecentroidr(i), linecentroidc(i), linecentroidr(j), linecentroidc(j), scale, linewidth, fontsz, "green", false);
        end
    end
    img = insertText(img, [0,0], "various relations", TextColor="white", BoxColor="black", FontSize=fontsz);
    img = insertText(img, [0,22*scale], "¢Face", TextColor="green", BoxColor="black", FontSize=fontsz);
end


function img = RenderCLEVRPosImgAfterAbduction(prototype, img, scale, fontsz, linewidth, is_color_mismatch, randomized_obj_idx)
    % tier 1, after abduction

    is_twoeyes_randomized = (randomized_obj_idx == 1 || randomized_obj_idx == 2);
    is_eyesnose_randomized = (randomized_obj_idx == 1 || randomized_obj_idx == 2 || randomized_obj_idx== 3);
    is_mouth_randomized = (randomized_obj_idx == 4 || randomized_obj_idx == 5 || randomized_obj_idx == 6);
    
    edgemask = ((prototype.obj1idx == 1) & (prototype.obj2idx == 2)) | ((prototype.obj1idx == 2) & (prototype.obj2idx == 1));
    img = RenderLine(img, prototype.obj1r(edgemask), prototype.obj1c(edgemask), prototype.obj2r(edgemask), prototype.obj2c(edgemask), scale, linewidth, fontsz, "yellow", is_twoeyes_randomized || is_color_mismatch);
    
    edgemask = ((prototype.obj1idx == 1) & (prototype.obj2idx == 3)) | ((prototype.obj1idx == 3) & (prototype.obj2idx == 1));
    img = RenderLine(img, prototype.obj1r(edgemask), prototype.obj1c(edgemask), prototype.obj2r(edgemask), prototype.obj2c(edgemask), scale, linewidth, fontsz, "cyan", is_eyesnose_randomized);
    
    edgemask = ((prototype.obj1idx == 2) & (prototype.obj2idx == 3)) | ((prototype.obj1idx == 3) & (prototype.obj2idx == 2));
    img = RenderLine(img, prototype.obj1r(edgemask), prototype.obj1c(edgemask), prototype.obj2r(edgemask), prototype.obj2c(edgemask), scale, linewidth, fontsz, "cyan", is_eyesnose_randomized);
    
    edgemask = ((prototype.obj1idx == 4) & (prototype.obj2idx == 5)) | ((prototype.obj1idx == 5) & (prototype.obj2idx == 4));
    img = RenderLine(img, prototype.obj1r(edgemask), prototype.obj1c(edgemask), prototype.obj2r(edgemask), prototype.obj2c(edgemask), scale, linewidth, fontsz, "magenta", is_mouth_randomized);
    
    edgemask = ((prototype.obj1idx == 5) & (prototype.obj2idx == 6)) | ((prototype.obj1idx == 6) & (prototype.obj2idx == 5));
    img = RenderLine(img, prototype.obj1r(edgemask), prototype.obj1c(edgemask), prototype.obj2r(edgemask), prototype.obj2c(edgemask), scale, linewidth, fontsz, "magenta", is_mouth_randomized);

    % tier 2, after abduction

    edgemask = ((prototype.obj1idx == 1) & (prototype.obj2idx == 2)) | ((prototype.obj1idx == 2) & (prototype.obj2idx == 1));
    eyecentroidr = prototype.obj1r(edgemask) + (prototype.obj2r(edgemask) - prototype.obj1r(edgemask)) / 2;
    eyecentroidc = prototype.obj1c(edgemask) + (prototype.obj2c(edgemask) - prototype.obj1c(edgemask)) / 2;
    
    nosecentroidr = cat(1, prototype.obj1r(prototype.obj1idx == 3), prototype.obj2r(prototype.obj2idx == 3));
    nosecentroidc = cat(1, prototype.obj1c(prototype.obj1idx == 3), prototype.obj2c(prototype.obj2idx == 3));

    mouthcentroidr = cat(1, prototype.obj1r(prototype.obj1idx == 5), prototype.obj2r(prototype.obj2idx == 5));
    mouthcentroidc = cat(1, prototype.obj1c(prototype.obj1idx == 5), prototype.obj2c(prototype.obj2idx == 5));

    img = RenderLine(img, eyecentroidr(1), eyecentroidc(1), nosecentroidr(1), nosecentroidc(1), scale, linewidth, fontsz, "green", ~is_color_mismatch);
    img = RenderLine(img, nosecentroidr(1), nosecentroidc(1), mouthcentroidr(1), mouthcentroidc(1), scale, linewidth, fontsz, "green", ~is_color_mismatch);
end


function img = RenderLine(img, obj1r, obj1c, obj2r, obj2c, scale, linewidth, fontsz, color, do_render_x)
    arguments
        img, obj1r(1,1), obj1c(1,1), obj2r(1,1), obj2c(1,1), scale(1,1), linewidth(1,1), fontsz, color, do_render_x(1,1)
    end
    img = insertShape(img, "line", scale.*[obj1r,obj1c,obj2r,obj2c], Color=color, LineWidth=linewidth);
    
    if do_render_x
        centroidr = obj1r + (obj2r - obj1r) / 2;
        centroidc = obj1c + (obj2c - obj1c) / 2;
        img = insertText(img, scale.*[centroidr,centroidc], "X", TextColor="red", BoxOpacity=0, AnchorPoint="Center", FontSize=1.5*fontsz, Font="Leelawadee UI Bold");
    end
end