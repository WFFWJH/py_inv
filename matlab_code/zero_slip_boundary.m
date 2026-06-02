function [W,d] = zero_slip_boundary(slip_model,segment_ID,top_layer_no,ratio,ramp_choice,varargin)
% add zero-slip constraint at the top surface
% Supplements: add zero-slip constraint at the bottom surface
%           also add zero-slip slip boundary slip at one side of the fault
% options: 'layer_no': add zeros slip for a certain layer (usually top and bottom layer)
%          'left': left side of the fault plane (for patches of the whole layer)
%          'right': right side of the fault plane

Np = size(slip_model,1);
all_fault_id = slip_model(:,1);
all_patch_id = slip_model(:,2);
all_layer_id = slip_model(:,3);
nL = compute_patch_each_layer(slip_model);

d = zeros(2*Np,1);
dip_smooth = true;
if ~isempty(varargin)
    for CC = 1:floor(length(varargin)/2)
        try
            switch lower(varargin{CC*2-1})

                case 'dip_smooth'
                    dip_smooth = varargin{CC*2};
            end
        catch
            error('Unrecognized Keyword');
        end
    end
end
% select this fault segment
NF = length(segment_ID);
 V = zeros(2*Np,1);
    if ~dip_smooth
       V = zeros(Np,1);
       d = zeros(Np,1);
    end

for ii = 1:NF
    this_segment_ID = segment_ID(ii);
    patch_this_segment = all_patch_id(all_fault_id==this_segment_ID);
    layer_this_segment = all_layer_id(all_fault_id==this_segment_ID);
    nL_this_segment = nL(this_segment_ID,:);
    patch_before_this_segment = sum(sum(nL(1:this_segment_ID-1,:)));

    if isnumeric(top_layer_no)
        patch_top_this_segment = patch_this_segment(layer_this_segment == top_layer_no);
    elseif strcmp(top_layer_no,'left')      % start from the first patch of each layer
        Nlayer = max(layer_this_segment);
        patch_top_this_segment = zeros(Nlayer,1);
        for jj = 1:Nlayer
            patch_top_this_segment(jj) = patch_before_this_segment + 1 + sum(nL_this_segment(1:jj-1));
        end
    elseif strcmp(top_layer_no,'right')
        Nlayer = max(layer_this_segment);
        patch_top_this_segment = zeros(Nlayer,1);
        for jj = 1:Nlayer
            patch_top_this_segment(jj) = patch_before_this_segment + sum(nL_this_segment(1:jj));
        end
    else
        error('There is something wrong with the patch options!');
    end

    strike_indx = patch_top_this_segment';
    %     disp(strike_indx);

    dip_indx = strike_indx + Np;
    %     disp(dip_indx);
    if dip_smooth
        zero_slip_indx = [strike_indx,dip_indx];
    else
        zero_slip_indx = [strike_indx];
    end
    V(zero_slip_indx) = ratio;
end
W = diag(V);      % Use diagonal function to speed it up

h1 = size(W,1);
ramp_choice = lower(ramp_choice);
if strcmp(ramp_choice,'bi_ramp')
    rmp = zeros(h1,4);       % bilinear ramp
elseif strcmp(ramp_choice,'qu_ramp_7')
    rmp = zeros(h1,7);
elseif strcmp(ramp_choice,'qu_ramp_5')
    rmp = zeros(h1,5);
else
    rmp = [];
end
W = [W,rmp];

end