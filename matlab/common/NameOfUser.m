% Eli Bowen 5/21/2021
% RETURNS
%   name - the username of whoever's running matlab
function name = NameOfUser()
    if ispc()
        userProfile = getenv('USERPROFILE');
        [p,name,ext] = fileparts(userProfile); % get last folder in the above path
        name = strcat(name, ext); % in case the folder name has a dot in it
    else
        [status,name] = system('id -un'); % def works on centos
        name = deblank(name); % remove newline
    end
end