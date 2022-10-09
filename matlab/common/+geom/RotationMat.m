% Eli Bowen 12/19/2020
% INPUTS
%   r                     - (numeric) a 2D or 3D rotation formatted as [r] or [rX,rY,rZ]
%   angleFormat           - (char) 'deg' | 'rad' (is r in units of degrees or radians?)
%   do_homogeneous_coords - scalar (logical) default = false
% RETURNS
%   rotation - a rotation matrix (if 3D, performs rotations in order x, y, z)
function rotation = RotationMat(r, angleFormat, do_homogeneous_coords)
    validateattributes(r, {'numeric'}, {'nonempty','vector'}, 1);
    validateattributes(angleFormat, {'char'}, {'nonempty'}, 2);
    if ~exist('do_homogeneous_coords', 'var') || isempty(do_homogeneous_coords)
        do_homogeneous_coords = false;
    end

    if numel(r) == 1
        if strcmp(angleFormat, 'deg')
            if do_homogeneous_coords
                rotation = [cosd(r),-sind(r),0;...
                            sind(r),cosd(r),0;...
                            0,0,1];
            else
                rotation = [cosd(r),-sind(r);...
                            sind(r),cosd(r)];
            end
        elseif strcmp(angleFormat, 'rad')
            if do_homogeneous_coords
                rotation = [cos(r),-sin(r),0;...
                            sin(r),cos(r),0;...
                            0,0,1];
            else
                rotation = [cos(r),-sin(r);...
                            sin(r),cos(r)];
            end
        else
            error('unknown input angleFormat');
        end
    elseif numel(r) == 3
        if strcmp(angleFormat, 'deg')
            if do_homogeneous_coords
                error('TODO');
            else
                rxMat = [1,0,0;...
                         0,cosd(r(1)),-sind(r(1));...
                         0,sind(r(1)),cosd(r(1))];
                ryMat = [cosd(r(2)),0,sind(r(2));...
                         0,1,0;...
                         -sind(r(2)),0,cosd(r(2))];
                rzMat = [cosd(r(3)),-sind(r(3)),0;...
                         sind(r(3)),cosd(r(3)),0;...
                         0,0,1];
            end
        elseif strcmp(angleFormat, 'rad')
            if do_homogeneous_coords
                error('TODO');
            else
                rxMat = [1,0,0;...
                         0,cos(r(1)),-sin(r(1));...
                         0,sin(r(1)),cos(r(1))];
                ryMat = [cos(r(2)),0,sin(r(2));...
                         0,1,0;...
                         -sin(r(2)),0,cos(r(2))];
                rzMat = [cos(r(3)),-sin(r(3)),0;...
                         sin(r(3)),cos(r(3)),0;...
                         0,0,1];
            end
        else
            error('unknown input angleFormat');
        end

        rotation = rxMat * ryMat * rzMat;
    else
        error('r must be 2D (1 angle) or 3D (3 angles)');
    end
end