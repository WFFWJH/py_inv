function [slip_model,RMS_misfit,model_roughness,return_var] = make_fault_from_insar1(slip_model_vs,slip_model_ds,iter_step,tracks,data_type,varargin)
% Build the finite fault model using fault trace derived from both offsets and seismicity data
% return the variance reduction between the model and data
% Started by Zeyu Jin on 07/15/2019
% added data_list input later for experiment. Xiaoyu Zou, 11/2/2022
format long
%% default values
lambda = 1;%default 1e-1

alpha=0.8;%weight to the interferogram, default 1
beta=1;%weight to the offsets data, defaut 0.2
% gpsweight=0.3; %weight to the GNSS data, default 0.3
% alpha = 0.25;  % relative weight of RNG data default 0.25
% beta = 1;   % relative weight of ALOS-2 data default 1
% gamma = 0.2;  % relative weight of AZO data (from CSK) default 0.2
% gpsweight=2;%relative weight of GPS data default 0.3
segment_file = [];    intersect_file = [];
shallow_dip_id = [];
model_type = 'okada';

Con=[0 0 0];%Sign constraint; put 1 for positivity, -1 for negativity, 0 for no constraint

%% read varargin values and assembly
if ~isempty(varargin)
    for CC = 1:floor(length(varargin)/2)
        try
            switch lower(varargin{CC*2-1})
                case 'smoothness'
                    lambda = varargin{CC*2};
                case 'rng_ratio'
                    alpha = varargin{CC*2};
                case 'alos_ratio'
                    beta = varargin{CC*2};
                case 'azo_ratio'
                    gamma = varargin{CC*2};
                case 'segment_smooth_file'
                    segment_file = varargin{CC*2};
                case 'intersect_smooth_file'
                    intersect_file = varargin{CC*2};
                case 'shallow_dip_id'
                    shallow_dip_id = varargin{CC*2};  % to control the dip slip component
                case 'model_type'
                    model_type = varargin{CC*2};  % homogenous or layered
                case 'fault'
                    fault_file = varargin{CC*2};
                case 'ref_lon'
                    ref_lon = varargin{CC*2};
                case 'lonc'
                    lon_eq = varargin{CC*2};
                case 'latc'
                    lat_eq = varargin{CC*2};
                case 'con'
                    Con = varargin{CC*2};
                case 'ramp'
                    ramp_choice = varargin{CC*2};
            end
        catch
            error('Unrecognized Keyword');
        end
    end
end

%% read downsampled data
iint = iter_step;  % the number of resampling iteration
slip_model = [slip_model_vs;slip_model_ds];
slip_model(:,2)=[1:size(slip_model,1)]';    % recomputed finally to combine all the fault segments
disp(['There are total ',num2str(max(slip_model(:,1))),'segments']);
%     fid = fopen(data_list);
%     tmp_txt = fgetl(fid);
%     while tmp_txt ~=-1
%         indx=find(tmp_txt=='/');
%         [G_raw,G,bd_raw,bd] = build_green_function(slip_model,[tmp_txt(indx(end-1)+1:indx(end)),'/LOS2/los_samp',num2str(iint),'.mat'],'insar','noramp',model_type);
%     end

% % [G1_raw,G1,bd1_raw,bd1] = build_green_function(slip_model,['../WORK1/azi_best/D33/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% % [G2_raw,G2,bd2_raw,bd2] = build_green_function(slip_model,['../WORK1/azi_best/D33/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% [G3_raw,G3,bd3_raw,bd3] = build_green_function(slip_model,['../WORK1/azi_best/D106/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);
% % [G4_raw,G4,bd4_raw,bd4] = build_green_function(slip_model,['../WORK1/azi_best/D106/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% % [G5_raw,G5,bd5_raw,bd5] = build_green_function(slip_model,['../WORK1/azi_best/D106/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% [G6_raw,G6,bd6_raw,bd6] = build_green_function(slip_model,['../WORK1/azi_best/D33/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);
% % [G7_raw,G7,bd7_raw,bd7] = build_green_function(slip_model,['../WORK1/azi_best/A143/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% % [G8_raw,G8,bd8_raw,bd8] = build_green_function(slip_model,['../WORK1/azi_best/A143/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% [G9_raw,G9,bd9_raw,bd9] = build_green_function(slip_model,['../WORK1/azi_best/A143/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);
% % [G10_raw,G10,bd10_raw,bd10] = build_green_function(slip_model,['../WORK1/azi_best/A70/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% % [G11_raw,G11,bd11_raw,bd11] = build_green_function(slip_model,['../WORK1/azi_best/A70/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
% [G12_raw,G12,bd12_raw,bd12] = build_green_function(slip_model,['../WORK1/azi_best/A70/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);

G_raw = cell(numel(tracks),1);
G     = cell(numel(tracks),1);
bd_raw = cell(numel(tracks),1);
bd    = cell(numel(tracks),1);

labels_tracks = zeros(numel(tracks),1);
for i = 1:numel(tracks)
    fname = [ tracks{i}, '/los_samp', num2str(iint), '.mat'];
    type = data_type{i};
    [G_raw{i}, G{i}, bd_raw{i}, bd{i}] = build_green_function( ...
        slip_model, fname, type, ramp_choice, model_type);
    if strcmp(type, 'AZO')
        G_raw{i}  = G_raw{i}  * beta;
        G{i}      = G{i}      * beta;
        bd_raw{i} = bd_raw{i} * beta;
        bd{i}     = bd{i}     * beta;
    elseif strcmp(type, 'insar')
        G_raw{i}  = G_raw{i}  * alpha;
        G{i}      = G{i}      * alpha;
        bd_raw{i} = bd_raw{i} * alpha;
        bd{i}     = bd{i}     * alpha;
    end

        % get track's number
    tok = regexp(tracks{i}, '/[AD](\d+)', 'tokens');
    if ~isempty(tok)
        labels_tracks(i) = str2double(tok{1}{1});   % 拿出第一匹配
    end
end

% same track's ramp should be same if used both insar and offset
unique_orbits = unique(labels_tracks);
class_map = zeros(size(tracks));
for i = 1:length(unique_orbits)
    class_map(labels_tracks == unique_orbits(i)) = i;
end


dip_smooth = true;
%% generate Green's function and smooth matrix
[H,h1,~] = build_smooth_function(slip_model_vs,slip_model_ds,segment_file,intersect_file,ramp_choice,'dip_id',shallow_dip_id,'dip_smooth',dip_smooth); %modified by xiaoyu


%% add zero-slip boundary for the fault bottom/left/right
plane_fault = 1:max(slip_model(:,1));
bottom_layer_no = max(slip_model(:,3)); ratio = 3e-4;
[Wb,db] = zero_slip_boundary(slip_model,plane_fault,bottom_layer_no,ratio,ramp_choice,'dip_smooth',dip_smooth);    % for the bottom layer


left_fault = [4];
ratio = 3e-4;
[Wr,dr] = zero_slip_boundary(slip_model,left_fault,'left',ratio,ramp_choice,'dip_smooth',dip_smooth);%used to be left, tested by xiaoyu to make it 'right'

right_fault = [1];   ratio = 3e-4;%tested by xiaoyu
[Wl,dl] = zero_slip_boundary(slip_model,right_fault,'right',ratio,ramp_choice,'dip_smooth',dip_smooth);%used to be right, tested by xiaoyu to make it 'left'
%%
if strcmp(ramp_choice, 'bi_ramp')
    add_col =4; % assume bilinear ramp
elseif strcmp(ramp_choice, 'qu_ramp_7')
    add_col=7;
elseif strcmp(ramp_choice, 'qu_ramp_5')
    add_col = 5;
else
    add_col = 0;
end


if add_col > 0

    G_raw_ramp = expand_last_columns(G_raw,add_col,'class_map',class_map);
    G_ramp = expand_last_columns(G,add_col,'class_map',class_map);

    matrices = {H,Wb,Wl,Wr};
    result1 = cell(size(matrices));
    for k = 1:length(matrices)
        A = matrices{k};
        B = zeros(size(A,1),3*add_col);
        C = zeros(size(A,1),size(A,2)+3*add_col);
        C(:,1:(end-add_col*3)) = A;
        C(:,(end-add_col*3+1):end) = B;
        result1{k} = C;
    end
    [H,Wb,Wl,Wr] = deal(result1{:});
end
%% construct the Green's function
% adjust the relative weight between ASC and DES track to be 1:1
% better for fitting one track and decomposition
%     A2D = 1/3;    % three ascending tracks (ASC64,T065,T066) & one descending track (DES71)
%     A2D = 1;       % equal weight of each dataset
%Testing by Xiaoyu Zou: changed all 's' to 'p'
nflt = max(slip_model(:,1));
tSm = zeros(1,nflt+1);
fault_id = slip_model(:,1);
for i=1:nflt
    tSm(i+1) = length(find(fault_id == i));
end


% G_raw = [alpha*G1;beta* G2;beta*G3;alpha*G4;beta*G5;beta*G6;alpha*G7;beta*G8;beta*G9;alpha*G10;beta*G11;beta*G12;Gs];
% % G_raw(:,sum(tSm)+1:end-4*add_col)=[];
% % H(:,sum(tSm)+1:end-4*add_col)=[];
% Greens = [G_raw;H*lambda/h1;Wb;Wl;Wr];
% 
% bd_raw = [alpha*bd1;beta*bd2;beta*bd3;alpha*bd4;beta*bd5;beta*bd6;alpha*bd7;beta*bd8;beta*bd9;alpha*bd10;beta*bd11;beta*bd12;bs];
% bdata_sm = [bd_raw;zeros(h1,1);db;dl;dr];
% 
% % G3_raw(:,sum(tSm)+1:end-4*add_col)=[];
% % G6_raw(:,sum(tSm)+1:end-4*add_col)=[];
% % G9_raw(:,sum(tSm)+1:end-4*add_col)=[];
% % G12_raw(:,sum(tSm)+1:end-4*add_col)=[];
% 
% GrF = [G1_raw;G2_raw;G3_raw;G4_raw;G5_raw;G6_raw;G7_raw;G8_raw;G9_raw;G10_raw;G11_raw;G12_raw;Gs_raw];
% Bdata = [bd1_raw;bd2_raw;bd3_raw;bd4_raw;bd5_raw;bd6_raw;bd7_raw;bd8_raw;bd9_raw;bd10_raw;bd11_raw;bd12_raw;bs_raw];

G_last = vertcat(G_ramp{:});
Greens = [G_last;H*lambda/h1;Wb;Wl;Wr];
bd_last = vertcat(bd{:});
bdata_sm = [bd_last;zeros(h1,1);db;dl;dr];
GrF = vertcat(G_raw_ramp{:});
Bdata = vertcat(bd_raw{:});

%% the postivity constraint (adapted from Yuri's code)

NT = 2; NS = nflt;  % the number of segments
%     [lb,ub] = bounds_new_C(NS,NT,tSm,add_col,slip_model);
%     [lb,ub] = bounds_new_B(NS,NT,tSm,add_col,slip_model);
%     [lb,ub] = bounds_new_A(NS,NT,tSm,add_col);
[lb,ub] = bounds_new(NS,NT,tSm,add_col,Con);
%     [lb,ub] = bounds_new_M5(NS,NT,tSm,add_col);
%     [lb,ub] = bounds_resolution(NS,NT,tSm,add_col);

% linear inversion
options = optimset('LargeScale','on','DiffMaxChange',1e-1,'DiffMinChange',1e-12, ...
    'TolCon',1e-12,'TolFun',1e-12,'TolPCG',1e-12,'TolX',1e-12,'MaxIter',1e2,'MaxPCGIter',1e9,'Diagnostics','on','Display','iter');
[u,resnorm,residual,exitflag] = lsqlin(Greens,double(bdata_sm),[],[],[],[],lb,ub,[],options);


% compute the reduction of total variance (before weighing) of the downsampled data
rms0 = sum(Bdata.^2);
rms1 = sum((GrF*u-Bdata).^2);
redu_perc = 100*(rms0-rms1)/rms0;
fprintf('rms misfit (dat., res.) = %e %e (%f%%) \n',rms0,rms1,redu_perc);
fprintf('RMS: %f\n',rms(GrF*u-Bdata));
fprintf('resnorm, resid. = %e %e \n',sqrt(resnorm),mean(residual));
fprintf('exitflag is %d\n',exitflag);   % 1 means the function converged to a solution x

rough_matrix = H*u;
RMS_misfit = sum((G_last*u - bd_last).^2);   % use chi-square statistic instead
model_roughness = sqrt(sum(rough_matrix.^2)/length(rough_matrix));


%% plot and save the finite fault inversion
slip_model(:,12) = u(1:sum(tSm));
slip_model(:,13) = u(sum(tSm)+1:end-4*add_col);


%% plot the resampled data fitting
insar_model = cell(numel(tracks),1);

for i = 1:numel(tracks)
    insar_model{i} = G_raw_ramp{i}*u;
end


if iint ==1
    save('greens_okada.mat', ...
         'G_last','Bdata','bd_last','slip_model', 'bdata_sm', 'GrF', ...
         'H', 'h1', 'Wb', 'Wl', 'Wr','ramp_choice');
    for i = 1:numel(tracks)
        fname = [ tracks{i}, '/los_samp', num2str(iint), '.mat'];
        plot_insar_model_resampled(fname,insar_model{i},'iter_step',iint,'fault',fault_file,'model_type',...
            model_type,'misfit_range',100,'defo_max',300,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
    end

end

% Temporary: also save a snapshot for iint==0 to allow Python-side verification.
% (Added for Python port comparison; can be removed when done.)
if iint == 0
    save('greens_okada_iint0.mat', ...
         'G_last','Bdata','bd_last','slip_model', 'bdata_sm', 'GrF', ...
         'H', 'h1', 'Wb', 'Wl', 'Wr','ramp_choice','u');
end
    
% compute_moment(slip_model,model_type);
mu = 30e9;
Apatch = slip_model(:,7).*slip_model(:,8);
strike_u = slip_model(:,12) ;     % in meters
strike_d = slip_model(:,13) ;
M0 = sum(mu .* sqrt(strike_u.^2 + strike_d.^2) .* Apatch);
Mw = 2/3*(log10(M0) - 9.1);
fprintf('The moment magnitude is Mw = %f\nM0 = %e\n',Mw,M0);

return_var = [redu_perc,exitflag,rms(GrF*u-Bdata),Mw,M0/1e20];
    
end  
