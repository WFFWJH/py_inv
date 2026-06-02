addpath ./matlab_code/  

lambda = 1;
    h1 = double(h1);
    bdata_sm = bdata_sm';
bd_last = bd_last';
Bdata = Bdata';
u1 = u';
%%
    % Greens = [G_raw;H*lambda/h1;Wb;Wl;Wr];
    Greens = [G_last;H*lambda/h1;Wb;Wl;Wr];
    % bdata_sm = [bd_last;zeros(h1,1);db;dl;dr];

    nflt = max(slip_model(:,1));
    tSm = zeros(1,nflt+1);
    fault_id = slip_model(:,1);
    for i=1:nflt
        tSm(i+1) = length(find(fault_id == i));
    end
    NT = 2; NS = nflt;  % the number of segments
    Con = [0 0 0];
    [lb,ub] = bounds_new(NS,NT,tSm,Con);
    %     [lb,ub] = bounds_new_M5(NS,NT,tSm,add_col);
    %     [lb,ub] = bounds_resolution(NS,NT,tSm,add_col);
if strcmp(ramp_choice, 'bi_ramp')
    add_col =4; % assume bilinear ramp
elseif strcmp(ramp_choice, 'qu_ramp_7')
    add_col=7;
elseif strcmp(ramp_choice, 'qu_ramp_5')
    add_col = 5;
else
    add_col = 0;
end


    % linear inversion
    options = optimset('LargeScale','on','DiffMaxChange',1e-1,'DiffMinChange',1e-12, ...
        'TolCon',1e-12,'TolFun',1e-12,'TolPCG',1e-12,'TolX',1e-12,'MaxIter',1e9,'MaxPCGIter',1e9,'Diagnostics','on','Display','iter');
    [u,resnorm,residual,exitflag] = lsqlin(Greens,double(bdata_sm),[],[],[],[],lb,ub,[],options);
    % [u,resnorm,residual,exitflag] = lsqlin(GrF,double(Bdata),[],[],[],[],lb,ub,[],options);
fprintf("u1-u max:%f\n",max(u1-u));

    % compute the reduction of total variance (before weighing) of the downsampled data
    rms0 = sum(Bdata.^2);
    rms1 = sum(sum((GrF*u-Bdata).^2));
    redu_perc = 100*(rms0-rms1)/rms0;
    fprintf('rms misfit (dat., res.) = %e %e (%f%%) \n',rms0,rms1,redu_perc);

    fprintf('resnorm, resid. = %e %e \n',sqrt(resnorm),mean(residual));
    fprintf('exitflag is %d\n',exitflag);   % 1 means the function converged to a solution x

    rough_matrix = H*u;
    RMS_misfit = sum((G_last*u - bd_last).^2);   % use chi-square statistic instead
    model_roughness = sqrt(sum(rough_matrix.^2)/length(rough_matrix));
    fprintf("RMS_misfit: %f , model_roughness: %f \n",RMS_misfit,model_roughness);
    slip_model1 = slip_model;
    slip_model1(:,12) = u(1:sum(tSm));
    slip_model1(:,13) = u(sum(tSm)+1:end-4*add_col);
    show_slip_model(slip_model1,'ref_lon',95,'lonc', 95.33,'latc', 19.61);
    show_slip_model_2d(slip_model1,'ref_lon',95,'lonc', 95.33,'latc', 19.61,'axis_range',[0 500 -30 0]);
    plot_slip_vs_depth_arr(slip_model1)

    RMS_1 = rms(GrF*u-Bdata);
    fprintf('RMS: %f\n',RMS_1);
    fprintf("max slip1: %f ,max slip2: %f, max slip: %f\n",max(abs(slip_model1(:,12))),max(abs(slip_model1(:,13))),max(sqrt(slip_model1(:,12).^2+slip_model1(:,13).^2)));
    compute_moment(slip_model1);
    %% strike no dip
    lambda = 1;
    Greens = [G_last;H*lambda/h1;Wb;Wl;Wr];


    nflt = max(slip_model(:,1));
    tSm = zeros(1,nflt+1);
    fault_id = slip_model(:,1);
    for i=1:nflt
        tSm(i+1) = length(find(fault_id == i));
    end
    NT = 2; NS = nflt;  % the number of segments
    Con = [0 0 0];
    [lb,ub] = bounds_new(NS,NT,tSm,Con);
    %     [lb,ub] = bounds_new_M5(NS,NT,tSm,add_col);
    %     [lb,ub] = bounds_resolution(NS,NT,tSm,add_col);
if strcmp(ramp_choice, 'bi_ramp')
    add_col =4; % assume bilinear ramp
elseif strcmp(ramp_choice, 'qu_ramp_7')
    add_col=7;
elseif strcmp(ramp_choice, 'qu_ramp_5')
    add_col = 5;
else
    add_col = 0;
end

Greens(:,sum(tSm)+1:end-4*add_col)=[];
GrF(:,sum(tSm)+1:end-4*add_col)=[];
    % linear inversion
    options = optimset('LargeScale','on','DiffMaxChange',1e-1,'DiffMinChange',1e-12, ...
        'TolCon',1e-12,'TolFun',1e-12,'TolPCG',1e-12,'TolX',1e-12,'MaxIter',1e9,'MaxPCGIter',1e9);
    [u,resnorm,residual,exitflag] = lsqlin(Greens,double(bdata_sm),[],[],[],[],lb,ub,[],options);
    % [u,resnorm,residual,exitflag] = lsqlin(GrF,double(Bdata),[],[],[],[],lb,ub,[],options);


    % compute the reduction of total variance (before weighing) of the downsampled data
    rms0 = sum(Bdata.^2);
    rms1 = sum((GrF*u-Bdata).^2);
    redu_perc = 100*(rms0-rms1)/rms0;
    fprintf('rms misfit (dat., res.) = %e %e (%f%%) \n',rms0,rms1,redu_perc);

    fprintf('resnorm, resid. = %e %e \n',sqrt(resnorm),mean(residual));
    fprintf('exitflag is %d\n',exitflag);   % 1 means the function converged to a solution x

    rough_matrix = H*u;
    RMS_misfit = sum((G_last*u - bd_last).^2);   % use chi-square statistic instead
    model_roughness = sqrt(sum(rough_matrix.^2)/length(rough_matrix));
    fprintf("RMS_misfit: %f , model_roughness: %f \n",RMS_misfit,model_roughness);
    slip_model1 = slip_model;
    slip_model1(:,12) = u(1:sum(tSm));
    % slip_model1(:,13) = u(sum(tSm)+1:end-4*add_col);
    show_slip_model(slip_model1,'ref_lon',95,'lonc', 95.33,'latc', 19.61,'axis_range',[50 150 -180 350 -50 0]);


    RMS_1 = rms(GrF*u-Bdata);
    fprintf('RMS: %f\n',RMS_1);
    fprintf("max slip1: %f ,max slip2: %f, max slip: %f\n",max(abs(slip_model1(:,12))),max(abs(slip_model1(:,13))),max(sqrt(slip_model1(:,12).^2+slip_model1(:,13).^2)));
    compute_moment(slip_model1);
%%

lambdas = [0.1,0.5,0.8,1.2,1.5];
rmss = zeros(1,length(lambdas));
redu_percs = zeros(1, length(lambdas));
for ii=1:length(lambdas)
    lambda = lambdas(ii);
    Greens = [G_last;H*lambda/h1;Wb;Wl;Wr];


    nflt = max(slip_model(:,1));
    tSm = zeros(1,nflt+1);
    fault_id = slip_model(:,1);
    for i=1:nflt
        tSm(i+1) = length(find(fault_id == i));
    end
    NT = 2; NS = nflt;  % the number of segments
    Con = [0 0 0];
    [lb,ub] = bounds_new(NS,NT,tSm,Con);
    %     [lb,ub] = bounds_new_M5(NS,NT,tSm,add_col);
    %     [lb,ub] = bounds_resolution(NS,NT,tSm,add_col);


    % linear inversion
    options = optimset('LargeScale','on','DiffMaxChange',1e-1,'DiffMinChange',1e-12, ...
        'TolCon',1e-12,'TolFun',1e-12,'TolPCG',1e-12,'TolX',1e-12,'MaxIter',1e9,'MaxPCGIter',1e9);
    [u,resnorm,residual,exitflag] = lsqlin(Greens,double(bdata_sm),[],[],[],[],lb,ub,[],options);
    % [u,resnorm,residual,exitflag] = lsqlin(GrF,double(Bdata),[],[],[],[],lb,ub,[],options);


    % compute the reduction of total variance (before weighing) of the downsampled data
    rms0 = sum(Bdata.^2);
    rms1 = sum((GrF*u-Bdata).^2);
    redu_perc = 100*(rms0-rms1)/rms0;
    fprintf('rms misfit (dat., res.) = %e %e (%f%%) \n',rms0,rms1,redu_perc);

    fprintf('resnorm, resid. = %e %e \n',sqrt(resnorm),mean(residual));
    fprintf('exitflag is %d\n',exitflag);   % 1 means the function converged to a solution x

    rough_matrix = H*u;
    RMS_misfit = sum((G_last*u - bd_last).^2);   % use chi-square statistic instead
    model_roughness = sqrt(sum(rough_matrix.^2)/length(rough_matrix));
    fprintf("RMS_misfit: %f , model_roughness: %f \n",RMS_misfit,model_roughness);
    slip_model1 = slip_model;
    slip_model1(:,12) = u(1:sum(tSm));
    slip_model1(:,13) = u(sum(tSm)+1:end-4*add_col);

    RMS_1 = rms(GrF*u-Bdata);
    fprintf('RMS: %f\n',RMS_1);
    fprintf("max slip1: %f ,max slip2: %f, max slip: %f\n",max(slip_model1(:,12)),max(slip_model1(:,13)),max(sqrt(slip_model1(:,12).^2+slip_model1(:,13).^2)));
    compute_moment(slip_model1);

    rmss(ii) = RMS_1;
    redu_percs(ii) = redu_perc;
end

figure;
plot(log10(lambdas), rmss, '-o', 'LineWidth', 1.5);
xlabel('Smoothness(lg)');
ylabel('Rms residual(cm)');
title(' ');
grid on;

figure;
plot(log10(lambdas), redu_percs, '-o', 'LineWidth',1.5);
xlabel('Smoothness(lg)');
ylabel('redu_perc (%)');
title("");
grid on;


%%
track_num = 4;

G_comsol = G_raw_azi;
% G_index = 1;
% G_index1 = 1;
% for i = 1:track_num
%     G_comsol(:,G_index1:G_index1+patch_num(i)*2-1) = [G_raw_azi(:,G_index:G_index+patch_num(i)-1), G_raw_azi(:,G_index+sum(patch_num):G_index+patch_num(i)-1+sum(patch_num))];
%     G_index = G_index+patch_num(i);
%     G_index1 = G_index1+2*patch_num(i);
% end
% assert(G_index1-1 == size(G_raw_azi,2));
G_comsol = [G_comsol,GrF(:,end-track_num*add_col+1:end)];
assert(size(G_comsol,2) == size(G_raw,2))
los_samp_paths = [...
    ..."./myanmar/D33/LOS/los_samp1.mat",...
    ..."./myanmar/D33/RNG/los_samp1.mat",...
    "./azi_best/D106/AZI/los_samp1.mat",...
    ..."./myanmar/D106/LOS/los_samp1.mat",...
    ... "./myanmar/D106/RNG/los_samp1.mat",...
    "./azi_best/D33/AZI/los_samp1.mat",...
    ..."./myanmar/A143/LOS/los_samp1.mat",...
    ... "./myanmar/A143/RNG/los_samp1.mat",...
    "./azi_best/A143/AZI/los_samp1.mat",...
    ..."./myanmar/A70/LOS/los_samp1.mat",...
    ..."./myanmar/A70/RNG/los_samp1.mat",...
    "./azi_best/A70/AZI/los_samp1.mat"];
points_every_track = zeros(4,1);
G_comsol_raw = G_comsol;
G_raw_index = 1;
d_comsol_raw = d_raw;
for i = 1:length(los_samp_paths)
    data = load(los_samp_paths(i));
    losin = data.sampled_insar_data(:,3);
    disp(length(losin));
    points_every_track(i) = length(losin);
    G_comsol_raw(G_raw_index:G_raw_index+length(losin)-1,:) = G_comsol_raw(G_raw_index:G_raw_index+length(losin)-1,:)./length(losin);
    d_comsol_raw(G_raw_index:G_raw_index+length(losin)-1) = d_comsol_raw(G_raw_index:G_raw_index+length(losin)-1)./length(losin);
    G_raw_index = G_raw_index+length(losin);
end
assert(sum(points_every_track) == size(G_comsol,1))

lambda = 1;
Greens1 = [G_comsol_raw;H*lambda/h1;Wb;Wl;Wr];
bdata_sm1 = [d_comsol_raw;zeros(size(Greens1,1)-size(d_comsol_raw,1),1)];


nflt = max(slip_model(:,1));
tSm = zeros(1,nflt+1);
fault_id = slip_model(:,1);
for i=1:nflt
    tSm(i+1) = length(find(fault_id == i));
end
NT = 2; NS = nflt;  % the number of segments
Con = [0 0 0];
[lb,ub] = bounds_new(NS,NT,tSm,Con);
%     [lb,ub] = bounds_new_M5(NS,NT,tSm,add_col);
%     [lb,ub] = bounds_resolution(NS,NT,tSm,add_col);


% linear inversion
options = optimset('LargeScale','on','DiffMaxChange',1e-1,'DiffMinChange',1e-12, ...
    'TolCon',1e-12,'TolFun',1e-12,'TolPCG',1e-12,'TolX',1e-12,'MaxIter',1e9,'MaxPCGIter',1e9,'Diagnostics','on','Display','iter');
[u1,resnorm,residual,exitflag] = lsqlin(Greens1,double(bdata_sm1),[],[],[],[],lb,ub,[],options);
% [u,resnorm,residual,exitflag] = lsqlin(GrF,double(d_raw),[],[],[],[],lb,ub,[],options);


% compute the reduction of total variance (before weighing) of the downsampled data
rms0 = sum(d_raw.^2);
rms1 = sum((G_comsol*u1-d_raw).^2);
redu_perc = 100*(rms0-rms1)/rms0;
fprintf('rms misfit (dat., res.) = %e %e (%f%%) \n',rms0,rms1,redu_perc);

fprintf('resnorm, resid. = %e %e \n',sqrt(resnorm),mean(residual));
fprintf('exitflag is %d\n',exitflag);   % 1 means the function converged to a solution x

rough_matrix = H*u1;
RMS_misfit = sum((G_comsol_raw*u1 - d_comsol_raw).^2);   % use chi-square statistic instead
model_roughness = sqrt(sum(rough_matrix.^2)/length(rough_matrix));
fprintf("RMS_misfit: %f , model_roughness: %f \n",RMS_misfit,model_roughness);
slip_model2 = slip_model;
slip_model2(:,12) = u1(1:sum(tSm));
slip_model2(:,13) = u1(sum(tSm)+1:end-4*add_col);
show_slip_model(slip_model2,'ref_lon',95,'lonc', 95.33,'latc', 19.61,'axis_range',[50 150 -180 350 -50 0]);

fprintf("max slip1: %f ,max slip2: %f, max slip: %f\n",max(slip_model2(:,12)),max(slip_model2(:,13)),max(sqrt(slip_model2(:,12).^2+slip_model2(:,13).^2)));
compute_moment(slip_model2);

%%
% los_samp_paths = [...
%     ..."./myanmar/D33/LOS/los_samp1.mat",...
%  "./myanmar/D33/RNG/los_samp1.mat",...
% "./myanmar/D33/AZI/los_samp1.mat",...
% ..."./myanmar/D106/LOS/los_samp1.mat",...
%  "./myanmar/D106/RNG/los_samp1.mat",...
% "./myanmar/D106/AZI/los_samp1.mat",...
% ..."./myanmar/A143/LOS/los_samp1.mat",...
%  "./myanmar/A143/RNG/los_samp1.mat",...
% "./myanmar/A143/AZI/los_samp1.mat",...
% ..."./myanmar/A70/LOS/los_samp1.mat",...
%  "./myanmar/A70/RNG/los_samp1.mat",...
% "./myanmar/A70/AZI/los_samp1.mat"];

los_samp_paths = [...
    ..."./myanmar/D33/LOS/los_samp1.mat",...
 ..."./myanmar/D33/RNG/los_samp1.mat",...
"./azi_best/D33/AZI/los_samp1.mat",...
..."./myanmar/D106/LOS/los_samp1.mat",...
... "./myanmar/D106/RNG/los_samp1.mat",...
"./azi_best/D106/AZI/los_samp1.mat",...
..."./myanmar/A143/LOS/los_samp1.mat",...
... "./myanmar/A143/RNG/los_samp1.mat",...
"./azi_best/A143/AZI/los_samp1.mat",...
..."./myanmar/A70/LOS/los_samp1.mat",...
 ..."./myanmar/A70/RNG/los_samp1.mat",...
"./azi_best/A70/AZI/los_samp1.mat"];
G_raw_index = 1;
for i = 1:length(los_samp_paths)
    data = load(los_samp_paths(i));
    losin = data.sampled_insar_data(:,3);
    disp(length(losin));
    xin = data.sampled_insar_data(:,1) / 1000;
    yin = data.sampled_insar_data(:,2) / 1000;
    plot_insar_model_resampled(los_samp_paths(i),xin,yin,losin,GrF(G_raw_index:G_raw_index+length(xin)-1,:)*u,'misfit_range',100-70*mod(i,2),'defo_max',300-200*mod(i,2));
    G_raw_index = G_raw_index+length(xin);
end

show_slip_model(slip_model1,'ref_lon',95,'lonc', 95.33,'latc', 19.61,'axis_range',[50 150 -180 350 -50 0]);
% show_slip_model_2d(slip_model1,'ref_lon',95,'lonc', 95.33,'latc', 19.61,'axis_range',[50 150 -180 350 -50 0]);



function [lb,ub]=bounds_new(NS,NT,tSm,Con)
slip_max = 10e2;
Npatch = sum(tSm);
lb=-slip_max*ones(NT*Npatch,1);  %lower bound in cm
ub= slip_max*ones(NT*Npatch,1);  %upper bound
lb(Npatch+1:2*Npatch) = -slip_max;   % dominated by strike slip
ub(Npatch+1:2*Npatch) = slip_max;

for i=1:NS
    k1=sum(tSm(1:i))+1;
    k2=sum(tSm(1:i+1));
    
    for k=k1:k2
        if Con(1) > 0, lb(k) = 0; end
        if Con(2) > 0, lb(k+Npatch) = 0; end
        if Con(1) < 0, ub(k) = 0; end
        if Con(2) < 0, ub(k+Npatch) = 0; end

    end
end
% 
% lb((NT*sum(tSm)+1):(NT*sum(tSm)+add_col),1)=-Inf;
% ub((NT*sum(tSm)+1):(NT*sum(tSm)+add_col),1)=Inf;

end


function compute_moment(slip_model)
% compute the scalar seismic moment
strike_u = slip_model(:,12);     % in meters
strike_d = slip_model(:,13);
D = sqrt(strike_u.^2 + strike_d.^2);
lpatch = slip_model(:,7);
wpatch = slip_model(:,8);
Apatch = lpatch .* wpatch;
mu = 33e9;
M0 = sum(mu .* D .* Apatch);
Mw = 2/3*(log10(M0) - 9.1);
disp(['The moment magnitude is Mw = \nM0 = ',num2str(Mw),num2str(M0)]);
fprintf('\n');
end



function plot_insar_model_resampled(sampled_data_file,xin,yin,losin,los_model,varargin)

   los_res = losin - los_model;

   
   [filepath,save_name,~] = fileparts(sampled_data_file);
   % indx = find(filepath == '/');
   label_name = filepath;
   
   % if ~isempty(indx)
   %     if length(indx) == 2
   %        label_name = filepath(1:indx(2)-1); 
   %     else
   %        label_name = filepath(1:indx(1)-1); 
   %     end
   %     save_name = filepath(1:indx(1)-1); 
   % end
   
   defo_max = 300; % cmax(losin);
   defo_min = -300; % min(losin);
   res_max = 300;
   iter_step = 0;
   fault_file = [];
   lon_eq = 95.33; 
   lat_eq = 19.61;
   ref_lon = 95;
   
   if ~isempty(varargin)
       for CC = 1:floor(length(varargin)/2)
           try
               switch lower(varargin{CC*2-1})
                   case 'misfit_range'
                       res_max = varargin{CC*2};
                   case 'defo_max'
                       defo_max = varargin{CC*2};
                       defo_min = -defo_max;
                   case 'iter_step'
                       iter_step = varargin{CC*2};
                   case 'fault'
                       fault_file = varargin{CC*2};
                   case 'axis_range'
                       axis_range = varargin{CC*2};
                       if length(axis_range) ~= 4
                           error('Something wrong with axis range');
                       end
                   case 'ref_lon'
                       ref_lon = varargin{CC*2};
                   case 'lonc'
                       lon_eq = varargin{CC*2};
                   case 'latc'
                       lat_eq = varargin{CC*2};
               end
           catch
               error('Unrecognized Keyword');
           end
       end
   end
   
   if ~isempty(fault_file)
       fault_trace = load(fault_file);
       lonf = [fault_trace(:,1);fault_trace(:,3)];  
       latf = [fault_trace(:,2);fault_trace(:,4)];
       LS = length(lonf) / 2;
   else
       fault_trace = load('./fault');
       lonf = [fault_trace(:,1);fault_trace(:,3)];  
       latf = [fault_trace(:,2);fault_trace(:,4)];
       LS = length(lonf) / 2;
   end
   
%    [xo,yo] = utm2ll(lon_eq,lat_eq,0,1);
   [xo,yo] = ll2xy(lon_eq,lat_eq,ref_lon);
   
   sz = 30;
%    h0=figure('units','normalized','outerposition',[0 0 1 1]);
%    set(h0,'renderer','painters');
   h0 = figure;
   
   subplot('Position',[0.04 0.55 0.42 0.42]); hold on
   scatter(xin,yin,sz,losin,'filled');
   % read_plot_fault_segment('SKFS_fault.txt','m','ll');
   colormap jet
   colorbar
   title(['Sampled Data (',label_name,')']);
   set(gca,'Fontsize',20);
   clim([defo_min defo_max]);
   % axis(axis_range);
   axis equal tight;
   % plot the fault segments
   if ~isempty(fault_trace)
       for ii = 1:LS
          slon = [lonf(ii) lonf(ii+LS)];
          slat = [latf(ii) latf(ii+LS)];
%         [xx,yy] = utm2ll(slon,slat,0,1);
          [xx,yy] = ll2xy(slon,slat,ref_lon);
          xs = (xx - xo) ./ 1000;
          ys = (yy - yo) ./ 1000;
          line(xs,ys,'color','black','linewidth',3);
       end
   end

   subplot('Position',[0.54 0.55 0.42 0.42]); hold on
   scatter(xin,yin,sz,los_model,'filled');
   % read_plot_fault_segment('SKFS_fault.txt','m','ll');
   colormap jet
   colorbar
   title('Model');
   set(gca,'Fontsize',20);
   clim([defo_min defo_max]);
   % axis(axis_range);
   axis equal tight;
   if ~isempty(fault_trace)
       for ii = 1:LS
          slon = [lonf(ii) lonf(ii+LS)];
          slat = [latf(ii) latf(ii+LS)];
%         [xx,yy] = utm2ll(slon,slat,0,1);
          [xx,yy] = ll2xy(slon,slat,ref_lon);
          xs = (xx - xo) ./ 1000;
          ys = (yy - yo) ./ 1000;
          line(xs,ys,'color','black','linewidth',3);
       end
   end

   subplot('Position',[0.26 0.03 0.42 0.42]); hold on
   scatter(xin,yin,sz,los_res,'filled');
   % read_plot_fault_segment('SKFS_fault.txt','m','ll');
   colormap jet
   colorbar
   title('Residual');
   set(gca,'Fontsize',20);
   clim([-res_max res_max]);       % center with zero
   % axis(axis_range);
   axis equal tight;
   if ~isempty(fault_trace)
       for ii = 1:LS
          slon = [lonf(ii) lonf(ii+LS)];
          slat = [latf(ii) latf(ii+LS)];
%           [xx,yy] = utm2ll(slon,slat,0,1);
          [xx,yy] = ll2xy(slon,slat,ref_lon);
          xs = (xx - xo) ./ 1000;
          ys = (yy - yo) ./ 1000;
          line(xs,ys,'color','black','linewidth',3);
       end
   end
   
       % 计算文字内容
      rms0 = sum(los_model.^2);
rms1 = sum(los_res.^2);
redu_perc = 100*(rms0-rms1)/rms0;
txt = sprintf('rms = %e\n misfit =  %e\n (dat., res.) =(%f%%)', rms0, rms1, redu_perc);

% 在整个 figure 的归一化坐标系里添加文字
annotation('textbox',[0.6,0.3,0.3,0.2], ...   % [x,y,w,h], 归一化坐标 (0~1)
           'String',txt, ...
           'FitBoxToText','on', ...
           'HorizontalAlignment','center', ...
           'VerticalAlignment','middle', ...
           'EdgeColor','none', ...
           'FontSize',14);
   
   set(h0,'PaperPositionMode','auto');
%    set(h0,'visible','off');

   fname = fullfile(char(filepath), sprintf('%s_misfit_%d', char(save_name), iter_step));
% 确保目录存在
if ~exist(char(filepath),'dir')
    mkdir(char(filepath));
end
saveas(h0, fname, 'epsc');
   
%    % save the residual
%    if iter_step == 3
%        residual = [xin,yin,los_res];
%        save([filepath,'/',save_name,'_residual.mat'],'residual');
%    end
end
