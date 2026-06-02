function slip_model = load_fault_one_plane(fault_segment_file, varargin)
% this function is used to build up multiple whole fault plane
% the fault plane can be vertical or dipping
% could not be used for shallow irregular dipping feature

d2r = pi / 180;
lon_eq = -117.5;
lat_eq = 35.5;
ref_lon = lon_eq;

%% Default geometric values for the vertical fault
fault_id = 0;
% divide fault patch versus depth
W = 30e3; % make the faults deeper
N_layer = 5;
lp_top = 2e3; % should be the similar spatial size with sampled data
bias_lp = 1.3;
bias_wp = 1.3;
zstart = 0;
dips = [];

%% read varargin values and assembly
if ~isempty(varargin)
    for CC = 1:floor(length(varargin)/2)
        try
            switch lower(varargin{CC*2-1})
                case 'lonc'
                    lon_eq = varargin{CC*2};
                case 'latc'
                    lat_eq = varargin{CC*2};
                case 'fault_id'
                    fault_id = varargin{CC*2}; % which segment start to count
                case 'width'
                    W = varargin{CC*2};
                case 'layers'
                    N_layer = varargin{CC*2};
                case 'len_top'
                    lp_top = varargin{CC*2};
                case 'l_ratio'
                    bias_lp = varargin{CC*2};
                case 'w_ratio'
                    bias_wp = varargin{CC*2};
                case 'depth_start'
                    zstart = varargin{CC*2};
                case 'dip'
                    dips = varargin{CC*2};
                case 'ref_lon'
                    ref_lon = varargin{CC*2};
            end
        catch
            error('Unrecognized Keyword\n');
        end
    end
end

%% read fault data
%     [xo,yo] = utm2ll(lon_eq,lat_eq,0,1);
[xo, yo] = ll2xy(lon_eq, lat_eq, ref_lon);
fault_data = load(fault_segment_file);
lon_pt = [fault_data(:, 1); fault_data(:, 3)];
lat_pt = [fault_data(:, 2); fault_data(:, 4)];
nflt = size(fault_data, 1);

%     [xutm_pt,yutm_pt]=utm2ll(lon_pt,lat_pt,0,1);  % use ll2xy.m instead
%     Convert coordinates of  faults to utm 
[xutm_pt, yutm_pt] = ll2xy(lon_pt, lat_pt, ref_lon);
xpt = xutm_pt - xo;
ypt = yutm_pt - yo;


%% the fault geometry is defined as Peter Shearer's textbook

wp_factor = zeros(N_layer, 1);
for k = 1:N_layer
    wp_factor(k) = bias_wp^(k - 1);
end
wp_top = W / sum(wp_factor);

wp_layer = zeros(N_layer, 1);

for j = 1:N_layer
    wp_layer(j) = wp_top * bias_wp^(j - 1);
end

strikes = zeros(nflt,1);
thetas = zeros(nflt,1);
xo_segments = zeros(nflt,1);
yo_segments = zeros(nflt,1);
N_per_layer=zeros(nflt,N_layer);
lp_this_layer = zeros(nflt,N_layer);
for i = 1:nflt
    xstart = xpt(i+nflt);
    ystart = ypt(i+nflt);
    xend = xpt(i);
    yend = ypt(i);
    dx = xend - xstart; % negative constraint (the fault starts from the bottom to the top)
    dy = yend - ystart;
    theta = atan2(dy, dx);
    [xo_segments(i), yo_segments(i)] = xy2XY(xstart, ystart, theta);
    strike1 = 90 - theta / d2r;
    if (strike1 < 0)
        strike1 = strike1 + 360;
    end
    thetas(i) = theta;
    strikes(i) = strike1;
    L = sqrt(dx^2+dy^2);
    for j = 1:N_layer
        lp_this_layer_rough = lp_top * bias_lp^(j - 1);
        N_this_layer = round(L/lp_this_layer_rough);
        if N_this_layer == 0
            N_this_layer = 1;
            lp_this_layer(i,j) = L;
        end
        N_per_layer(i,j) = N_this_layer;
        lp_this_layer(i,j) = L/N_this_layer;
    end
end

total_patches = sum(N_per_layer(:));
slip_model = zeros(total_patches,13);
index_all_patches = 0;  % just for loop
for i = 1:nflt
    
    fault_id = fault_id + 1;
    dip = dips(i);
    theta = thetas(i);
    xo_segment = xo_segments(i);
    yo_segment = yo_segments(i);

    indx_patch = 0; %% it will be recomputed finally to combine all the fault segments
    % dz_start=0;   % fault maybe in depth
    for j = 1:N_layer
        N_this_layer = N_per_layer(i,j);
        for k = 1:N_this_layer
            indx_patch = indx_patch + 1;
            indx_fault_this_layer = fault_id;
            indx_depth_this_layer = j;
            xpatch_tmp = xo_segment + (k - 1) * lp_this_layer(i,j);
            ypatch_tmp = yo_segment - cosd(dip) * (sum(wp_layer(1:j)) - wp_layer(j));
            [xp_this_layer, yp_this_layer] = xy2XY(xpatch_tmp, ypatch_tmp, -(theta));
            zp_this_layer = (-(sum(wp_layer(1:j)) - wp_layer(j))) * sind(dip) + zstart;
            lpatch_this_layer = lp_this_layer(i,j);
            wpatch_this_layer = wp_layer(j);
            strkp_this_layer = strikes(i);
            dip0_this_layer = dip;

            index_all_patches = index_all_patches+1;
            slip_model(index_all_patches,1:10) = [indx_fault_this_layer, indx_patch, indx_depth_this_layer, xp_this_layer, yp_this_layer, zp_this_layer, ...
                lpatch_this_layer, wpatch_this_layer, strkp_this_layer, dip0_this_layer];
            % slip_model(index_all_patches,:) = [indx_fault_this_layer, indx_patch, indx_depth_this_layer, xp_this_layer, yp_this_layer, zp_this_layer, ...
            %     lpatch_this_layer, wpatch_this_layer, strkp_this_layer, dip0_this_layer, tp_this_layer, slip1_this_layer, slip2_this_layer];
            
        end
    end
end
end