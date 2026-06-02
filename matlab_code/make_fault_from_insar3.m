function [slip_model,RMS_misfit,model_roughness] = make_fault_from_insar3(slip_model_vs,slip_model_ds,iter_step,varargin)
% Build the finite fault model using fault trace derived from both offsets and seismicity data
% return the variance reduction between the model and data
% Started by Zeyu Jin on 07/15/2019
% added data_list input later for experiment. Xiaoyu Zou, 11/2/2022

%% default values
lambda = 1;%default 1e-1

alpha=1;%weight to the interferogram, default 1
beta=1;%weight to the offsets data, defaut 0.2
% gpsweight=0.3; %weight to the GNSS data, default 0.3
% alpha = 0.25;  % relative weight of RNG data default 0.25
% beta = 1;   % relative weight of ALOS-2 data default 1
% gamma = 0.2;  % relative weight of AZO data (from CSK) default 0.2
% gpsweight=2;%relative weight of GPS data default 0.3
segment_file = [];    intersect_file = [];
shallow_dip_id = [];
model_type = 'okada';
fault_file = [];
ref_lon = 71;
lon_eq = -117.5;
lat_eq = 35.5;
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




% [G1_raw,G1,bd1_raw,bd1] = build_green_function(slip_model,['./myanmar/D33/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G2_raw,G2,bd2_raw,bd2] = build_green_function(slip_model,['./azi_best/D33/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G3_raw,G3,bd3_raw,bd3] = build_green_function(slip_model,['./azi_best/D33/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);
% [G4_raw,G4,bd4_raw,bd4] = build_green_function(slip_model,['./myanmar/D106/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G5_raw,G5,bd5_raw,bd5] = build_green_function(slip_model,['./azi_best/D106/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G6_raw,G6,bd6_raw,bd6] = build_green_function(slip_model,['./azi_best/D106/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);
% [G7_raw,G7,bd7_raw,bd7] = build_green_function(slip_model,['./myanmar/A143/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G8_raw,G8,bd8_raw,bd8] = build_green_function(slip_model,['./azi_best/A143/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G9_raw,G9,bd9_raw,bd9] = build_green_function(slip_model,['./azi_best/A143/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);

% [G10_raw,G10,bd10_raw,bd10] = build_green_function(slip_model,['./azi_best/A70/LOS/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G11_raw,G11,bd11_raw,bd11] = build_green_function(slip_model,['./azi_best/A70/RNG/los_samp',num2str(iint),'.mat'],'insar',ramp_choice,model_type);
[G12_raw,G12,bd12_raw,bd12] = build_green_function(slip_model,['./azi_best/A70/AZI/los_samp',num2str(iint),'.mat'],'AZO',ramp_choice,model_type);


G1_raw = []; G1 = []; bd1_raw = []; bd1 = [];
% G2_raw = []; G2 = []; bd2_raw = []; bd2 = [];
% G3_raw = []; G3 = []; bd3_raw = []; bd3 = [];
G4_raw = []; G4 = []; bd4_raw = []; bd4 = [];
% G5_raw = []; G5 = []; bd5_raw = []; bd5 = [];
% G6_raw = []; G6 = []; bd6_raw = []; bd6 = [];
G7_raw = []; G7 = []; bd7_raw = []; bd7 = [];
% G8_raw = []; G8 = []; bd8_raw = []; bd8 = [];
% G9_raw = []; G9 = []; bd9_raw = []; bd9 = [];
G10_raw = []; G10 = []; bd10_raw = []; bd10 = [];
% G11_raw = []; G11 = []; bd11_raw = []; bd11 = [];
% G12_raw = []; G12 = []; bd12_raw = []; bd12 = [];

   Gs_raw = []; Gs = []; bs_raw = []; bs = [];
%  Gp_raw = []; Gp = []; bp_raw = []; bp = [];
%     [G4_raw,G4,bd4_raw,bd4] = build_green_function(slip_model,['ALOS2_stripe/LOS/los_samp',num2str(iint),'.mat'],'insar','noramp',model_type);
%     [G5_raw,G5,bd5_raw,bd5] = build_green_function(slip_model,['ALOS2_stripe/MAI2/los_samp',num2str(iint),'.mat'],'AZO','noramp',model_type);
%[Gs_raw,Gs,bs_raw,bs] = build_green_function(slip_model,'/Users/xiaoyuzou/Library/CloudStorage/OneDrive-UCSanDiego/Research/TurkeyEQ/Dataclean/GPS/M7578_new3.mat','camp_gps','noramp',model_type,gpsweight);
    % [Gs_raw,Gs,bs_raw,bs] = build_green_function(slip_model,'GPS/survey_gps_2d.mat','camp_gps','noramp',model_type, 10);


%% generate Green's function and smooth matrix
[H,h1,~] = build_smooth_function(slip_model_vs,slip_model_ds,segment_file,intersect_file,ramp_choice,'dip_id',shallow_dip_id); %modified by xiaoyu


%% add zero-slip boundary for the fault bottom/left/right
plane_fault = 1:max(slip_model(:,1));
bottom_layer_no = max(slip_model(:,3)); ratio = 5e-4;
[Wb,db] = zero_slip_boundary(slip_model,plane_fault,bottom_layer_no,ratio,ramp_choice);    % for the bottom layer

% left_fault = [14,16,10,9];
% ratio = 3e-4;
% [Wl,dl] = zero_slip_boundary(slip_model,left_fault,'right',ratio);%used to be left, tested by xiaoyu to make it 'right'
% 
% right_fault = [11,1];   ratio = 3e-4;%tested by xiaoyu
% [Wr,dr] = zero_slip_boundary(slip_model,right_fault,'left',ratio);%used to be right, tested by xiaoyu to make it 'left'

left_fault = [1];
ratio = 3e-4;
[Wr,dr] = zero_slip_boundary(slip_model,left_fault,'right',ratio,ramp_choice);%used to be left, tested by xiaoyu to make it 'right'

right_fault = [4];   ratio = 3e-4;%tested by xiaoyu
[Wl,dl] = zero_slip_boundary(slip_model,right_fault,'left',ratio,ramp_choice);%used to be right, tested by xiaoyu to make it 'left'

if strcmp(ramp_choice, 'bi_ramp')
    add_col =4; % assume bilinear ramp
elseif strcmp(ramp_choice, 'qu_ramp_7')
    add_col=7;
elseif strcmp(ramp_choice, 'qu_ramp_5')
    add_col = 5;
else
    add_col = 0;
end

% --- 假定 add_col 已由上层代码设置好 ---
% add_col = 4; % 示例

if add_col > 0
    % ===== 用户需修改的部分：把你的矩阵按顺序放入 matrices 列表 =====
    matrices = {G2_raw, G3_raw, G5_raw, G6_raw, G8_raw, G9_raw,G11_raw, G12_raw}; % 示例：6 个矩阵
    % ===================================================================

    % ---------- 方案 A：按 class_sizes 顺序分配（推荐） ----------
    % class_sizes 的和必须等于 numel(matrices)
    % 例：4 类，分别包含 [2,1,2,1] 个矩阵 --> 总和 6
    class_sizes = [2, 2, 2, 2];
    % ---------------------------------------------------------------

    % ---------- 方案 B（可选）：显式为每个矩阵指定类 ----------
    % 如果你想显式映射而不是顺序分配，取消方案 A 的使用并用下面的 class_map
    % class_map = [1 1 2 3 3 4]; % 每个元素对应 matrices 中同索引矩阵的类号
    % n_classes = max(class_map);
    % ---------------------------------------------------------------

    % 下面自动根据选用方案构造 class_map（不需要同时使用 A 和 B）
    if exist('class_sizes','var') && ~isempty(class_sizes)
        if sum(class_sizes) ~= numel(matrices)
            error('sum(class_sizes) must equal number of matrices (%d).', numel(matrices));
        end
        % 构建 class_map（按顺序分配）
        class_map = zeros(1, numel(matrices));
        idx = 1;
        for cls = 1:numel(class_sizes)
            for t = 1:class_sizes(cls)
                class_map(idx) = cls;
                idx = idx + 1;
            end
        end
        n_classes = numel(class_sizes);
    elseif exist('class_map','var') && ~isempty(class_map)
        if numel(class_map) ~= numel(matrices)
            error('class_map length must equal number of matrices.');
        end
        n_classes = max(class_map);
    else
        error('Please provide either class_sizes or class_map.');
    end

    % --------- 处理每个矩阵，将最后 add_col 替换为 n_classes*add_col ---------
    n_m = numel(matrices);
    result = cell(size(matrices));
    for idx = 1:n_m
        A = matrices{idx};
        if size(A,2) < add_col
            error('Matrix %d has fewer columns (%d) than add_col (%d).', idx, size(A,2), add_col);
        end

        lastCols = A(:, end-add_col+1:end);         % 取出最后 add_col 列
        B = zeros(size(A,1), n_classes * add_col);  % 扩展块，宽度为 n_classes * add_col
        cls = class_map(idx);                       % 当前矩阵属于第 cls 类
        pos = (cls-1)*add_col + (1:add_col);        % 在 B 中应该放置的位置
        B(:, pos) = lastCols;                       % 填入

        ncolA = size(A,2);
        new_ncol = ncolA + (n_classes-1)*add_col;   % 新总列数
        C = zeros(size(A,1), new_ncol);
        % 复制原矩阵除最后 add_col 之外的部分
        C(:, 1:(ncolA - add_col)) = A(:, 1:(ncolA - add_col));
        % 将扩展块放到尾部
        C(:, (ncolA - add_col + 1):new_ncol) = B;

        result{idx} = C;
    end

    [G2_raw, G3_raw, G5_raw, G6_raw, G8_raw, G9_raw,G11_raw, G12_raw] = deal(result{:});
    % ===================================================================

    % matrices = {G1,G3,G4,G6,G7,G9,G10,G12};
    matrices = {G2, G3, G5, G6, G8, G9,G11, G12};
        class_sizes = [2, 2, 2, 2];
    % ---------------------------------------------------------------

    % ---------- 方案 B（可选）：显式为每个矩阵指定类 ----------
    % 如果你想显式映射而不是顺序分配，取消方案 A 的使用并用下面的 class_map
    % class_map = [1 1 2 3 3 4]; % 每个元素对应 matrices 中同索引矩阵的类号
    % n_classes = max(class_map);
    % ---------------------------------------------------------------

    % 下面自动根据选用方案构造 class_map（不需要同时使用 A 和 B）
    if exist('class_sizes','var') && ~isempty(class_sizes)
        if sum(class_sizes) ~= numel(matrices)
            error('sum(class_sizes) must equal number of matrices (%d).', numel(matrices));
        end
        % 构建 class_map（按顺序分配）
        class_map = zeros(1, numel(matrices));
        idx = 1;
        for cls = 1:numel(class_sizes)
            for t = 1:class_sizes(cls)
                class_map(idx) = cls;
                idx = idx + 1;
            end
        end
        n_classes = numel(class_sizes);
    elseif exist('class_map','var') && ~isempty(class_map)
        if numel(class_map) ~= numel(matrices)
            error('class_map length must equal number of matrices.');
        end
        n_classes = max(class_map);
    else
        error('Please provide either class_sizes or class_map.');
    end

    % --------- 处理每个矩阵，将最后 add_col 替换为 n_classes*add_col ---------
    n_m = numel(matrices);
    result = cell(size(matrices));
    for idx = 1:n_m
        A = matrices{idx};
        if size(A,2) < add_col
            error('Matrix %d has fewer columns (%d) than add_col (%d).', idx, size(A,2), add_col);
        end

        lastCols = A(:, end-add_col+1:end);         % 取出最后 add_col 列
        B = zeros(size(A,1), n_classes * add_col);  % 扩展块，宽度为 n_classes * add_col
        cls = class_map(idx);                       % 当前矩阵属于第 cls 类
        pos = (cls-1)*add_col + (1:add_col);        % 在 B 中应该放置的位置
        B(:, pos) = lastCols;                       % 填入

        ncolA = size(A,2);
        new_ncol = ncolA + (n_classes-1)*add_col;   % 新总列数
        C = zeros(size(A,1), new_ncol);
        % 复制原矩阵除最后 add_col 之外的部分
        C(:, 1:(ncolA - add_col)) = A(:, 1:(ncolA - add_col));
        % 将扩展块放到尾部
        C(:, (ncolA - add_col + 1):new_ncol) = B;

        result{idx} = C;
    end

    [G2, G3, G5, G6, G8, G9,G11, G12] = deal(result{:});

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
G_raw = [alpha*G1;beta* G2;beta*G3;alpha*G4;beta*G5;beta*G6;alpha*G7;beta*G8;beta*G9;alpha*G10;beta*G11;beta*G12;Gs];
Greens = [G_raw;H*lambda/h1;Wb;Wl;Wr];

bd_raw = [alpha*bd1;beta*bd2;beta*bd3;alpha*bd4;beta*bd5;beta*bd6;alpha*bd7;beta*bd8;beta*bd9;alpha*bd10;beta*bd11;beta*bd12;bs];
bdata_sm = [bd_raw;zeros(h1,1);db;dl;dr];

GrF = [G1_raw;G2_raw;G3_raw;G4_raw;G5_raw;G6_raw;G7_raw;G8_raw;G9_raw;G10_raw;G11_raw;G12_raw;Gs_raw];
Bdata = [bd1_raw;bd2_raw;bd3_raw;bd4_raw;bd5_raw;bd6_raw;bd7_raw;bd8_raw;bd9_raw;bd10_raw;bd11_raw;bd12_raw;bs_raw];

%% the postivity constraint (adapted from Yuri's code)
nflt = max(slip_model(:,1));
tSm = zeros(1,nflt+1);
fault_id = slip_model(:,1);
for i=1:nflt
    tSm(i+1) = length(find(fault_id == i));
end



NT = 2; NS = nflt;  % the number of segments
%     [lb,ub] = bounds_new_C(NS,NT,tSm,add_col,slip_model);
%     [lb,ub] = bounds_new_B(NS,NT,tSm,add_col,slip_model);
%     [lb,ub] = bounds_new_A(NS,NT,tSm,add_col);
[lb,ub] = bounds_new(NS,NT,tSm,add_col,Con);
%     [lb,ub] = bounds_new_M5(NS,NT,tSm,add_col);
%     [lb,ub] = bounds_resolution(NS,NT,tSm,add_col);

% save("gree.mat","bdata_sm","Greens","lb","ub");





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
RMS_misfit = sum((G_raw*u - bd_raw).^2);   % use chi-square statistic instead
model_roughness = sqrt(sum(rough_matrix.^2)/length(rough_matrix));


%% plot and save the finite fault inversion


slip_model(:,12) = u(1:sum(tSm));
slip_model(:,13) = u(sum(tSm)+1:end-4*add_col);
%     show_slip_model(slip_model,'seismicity_profile/Ross_seismicity_cut.mat');
%%    show_slip_model(slip_model,'misfit_range',400,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq,'axis_range',[60 110 -45 25 -25 0]);
%     show_slip_model(slip_model,'misfit_range',25,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq,'axis_range',[80 120 -10 40 -30 0]);
%     show_slip_model(slip_model,'misfit_range',25,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq,'axis_range',[80 120 -10 40 -12 0]);
%     show_slip_model(slip_model,'axis_range',[-25 0 15 45 -15 0],'misfit_range',10);
%     write_slip_model_okada(slip_model,'fault_M7.slip');
%     write_slip_model_okada(slip_model,'fault_M5.slip');



%% plot the resampled data fitting
% insar_model1 = G1_raw * u;
insar_model2 = G2_raw * u;
insar_model3 = G3_raw * u;
% insar_model4 = G4_raw * u;
insar_model5 = G5_raw * u;
insar_model6 = G6_raw * u;

% insar_model7 = G7_raw * u;
insar_model8 = G8_raw * u;
insar_model9 = G9_raw * u;
% insar_model10 = G10_raw * u;
insar_model11 = G11_raw * u;
insar_model12 = G12_raw * u;
% %     cgps_model = Gg_raw * u; 
% Floyd_gps_model = Gp_raw * u;
 %   survey_gps_model = Gs_raw * u;  
   %%

if iint ==1
save('greens_cache_no_ramp_rng_azi.mat', ...
     'G_raw','Bdata','bd_raw','slip_model', 'bdata_sm', 'GrF', ...
     'H', 'h1', 'Wb', 'Wl', 'Wr','ramp_choice', '-v7.3');

% plot_insar_model_resampled(['./myanmar/D33/LOS/los_samp',num2str(iint),'.mat'],insar_model1,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',20,'defo_max',100,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/D33/RNG/los_samp',num2str(iint),'.mat'],insar_model2,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',50,'defo_max',200,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/D33/AZI/los_samp',num2str(iint),'.mat'],insar_model3,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',100,'defo_max',300,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
% plot_insar_model_resampled(['./myanmar/D106/LOS/los_samp',num2str(iint),'.mat'],insar_model4,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',20,'defo_max',100,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/D106/RNG/los_samp',num2str(iint),'.mat'],insar_model5,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',50,'defo_max',200,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/D106/AZI/los_samp',num2str(iint),'.mat'],insar_model6,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',100,'defo_max',300,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
% plot_insar_model_resampled(['./myanmar/A143/LOS/los_samp',num2str(iint),'.mat'],insar_model7,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',20,'defo_max',100,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/A143/RNG/los_samp',num2str(iint),'.mat'],insar_model8,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',50,'defo_max',200,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/A143/AZI/los_samp',num2str(iint),'.mat'],insar_model9,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',100,'defo_max',300,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
% plot_insar_model_resampled(['./myanmar/A70/LOS/los_samp',num2str(iint),'.mat'],insar_model10,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',20,'defo_max',100,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/A70/RNG/los_samp',num2str(iint),'.mat'],insar_model11,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',50,'defo_max',200,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);
plot_insar_model_resampled(['./azi_best/A70/AZI/los_samp',num2str(iint),'.mat'],insar_model12,'iter_step',iint,'fault',fault_file,'model_type',model_type,'misfit_range',100,'defo_max',300,'ref_lon',ref_lon,'lonc',lon_eq,'latc',lat_eq);

end
    
 % 
    % verify GPS component
%     modelx = survey_gps_model(1:2)' * 10;  % to mm
%     modely = survey_gps_model(3:4)' * 10;
%     
%     disp(['The model predictions of North components (ALAI, ALA6) are ', num2str(modely), ' mm']);
%     disp('The observations of North components (ALAI, ALA6) are -11.6 and -17.6 mm');
%     disp(' ');
%     
%     disp(['The model predictions of East components (ALAI, ALA6) are ', num2str(modelx), ' mm']);
%     disp('The observations of East components (ALAI, ALA6) are 4.3 and 8.1 mm');
%     disp(' ');

%     % compute the moment contributed by M6 and M7 events individually    
%     all_indx = [1:size(slip_model,1)]';
%     segID = slip_model(:,1);
%     LL = [7 8];       % assume M6.4 slip concentrated on two LL faults
%     M6_indx = find(segID == LL(1) | segID == LL(2));
%     M7_indx = setdiff(all_indx,M6_indx);
%     M6_model = slip_model(M6_indx,:);
%     M7_model = slip_model(M7_indx,:);
%     compute_moment(M6_model,model_type);
%     compute_moment(M7_model,model_type);
%     compute_moment(slip_model,model_type);
    
%     model54 = slip_model(indx_M54,:); 
%     model58 = slip_model(indx_M58,:);
%     compute_moment(model54,model_type);
%     compute_moment(model58,model_type);
compute_moment(slip_model,model_type);
    
end  
