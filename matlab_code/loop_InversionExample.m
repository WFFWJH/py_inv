%% Geodetic Inversion Test
%11/15/2022, Xiaoyu Zou
clear
close all
profile on
% Step 0: set path and read in parameters & files

% setenv('PATH',[getenv('PATH'),':/opt/bin']);  % add the path of GMT
configfile='configfile.txt';
fid = fopen(configfile);
tmp_txt = fgetl(fid);
files=zeros(1,1);
files=string(files);
while tmp_txt ~=-1
    tmp_txt = fgetl(fid);
    skip=find(tmp_txt=='#');
    if ~isempty(skip)
        continue
    end
    files=[files tmp_txt];
end
files(1)=[];
files(end)=[];
fclose(fid);
% grdin=files(1);
% grdout=files(2);
data_list=files(3);  % LOS data
los_list=data_list;
fault_file=files(4);
segment_smooth_file = files(5);
segment_file = files(6);
intersect_smooth_file = [];
intersect_file=[];
slip_model_ds=[];%set empty for most cases, because it's used to construct the geometry of "Y shape" or "flower structure" formed by shallow splay faults.

% to find how many segments of fault
fid = fopen(fault_file);
tmp_txt = fgetl(fid);
nseg = 0;
while tmp_txt ~= -1
    nseg = nseg + 1;
    tmp_txt = fgetl(fid);
end
disp(['There are ',num2str(nseg),' segments of fault.']);
fclose(fid);


configpara='configpara.txt';
fid = fopen(configpara);
tmp_txt = fgetl(fid);
para=zeros(1,1);
para=string(para);
while tmp_txt ~=-1
    tmp_txt = fgetl(fid);
    skip=find(tmp_txt=='#');
    if ~isempty(skip)
        continue
    end
    para=[para tmp_txt];
end
para(1)=[];
para(end)=[];
para=str2double(para);
fclose(fid);
lonf=para(1);
latf=para(2);%coordinate of pixel point
ref_lon=para(3);
threshold=para(4);
lonc=para(5);
latc=para(6);%coordinate of reference point (0,0)
iter_step=para(7);
iter_step2=para(8);

dip_angle=[82 82 82 82 82];%the array of fault ids that have dip angles not equal to 90 degrees
ramp_choice = "qu_ramp_7";

% dip_angle = [62 88 91 97 ]
% dip_angle = [65 65 80 80 ]
% dip_angle = [60 60 80 100 ]
% Step 1: data cleaning using clean_insar-data. Remove some near-field

% Nmin = 10; Nmax = 100;
% Nmin = 4;Nmax = 50;
% Nmin = 15;Nmax = 15;

% to find how many tracks of data
fid = fopen(data_list);
tmp_txt = fgetl(fid);
ntrack = 0;
while tmp_txt ~= -1
    ntrack = ntrack + 1;
    tmp_txt = fgetl(fid);
end
disp(['There are ',num2str(ntrack),' tracks of data.']);
fclose(fid);

% read txt file again to find those tracks and specify each sample regions
track = cell(ntrack,1);   
npt = zeros(ntrack,1);  
region = zeros(ntrack,4);
data_types = cell(ntrack,1); 
Nmin = zeros(ntrack,1);
Nmax = zeros(ntrack,1);

for ii = 1:ntrack
    region(ii,:) = [95 97.52 15.17 24.05];
end
fid = fopen(data_list);
tmp_txt = fgetl(fid);
count = 0;
while tmp_txt ~= -1
    count = count + 1;
    strs = strsplit(tmp_txt);
    track(count) = cellstr(strs{1});
    npt(count) = str2double(strs{2});
    num_of_strs = size(strs,2);
    % replaced by the region defined in the data_list
    if num_of_strs >= 6, region(count,:) = str2double(strs(3:6)); end   
    data_types(count) = cellstr(strs{7});
    tmp_txt = fgetl(fid);
end
fclose(fid);

for i = 1:ntrack
    if strcmp(data_types(i),'insar')
        Nmin(i) = 8;
        Nmax(i) = 500;
    elseif strcmp(data_types(i),'azi')
        Nmin(i) = 4;
        Nmax(i) = 50;
    end
end

a = 60:10:150;

[A,B,C,D,E] = ndgrid(a,a,a,a,a);

dip_angles = [A(:), B(:), C(:), D(:),E(:)];
mask = dip_angles(:,1) < 90 & ...
       dip_angles(:,1) <= dip_angles(:,2) & ...
       dip_angles(:,2) <= dip_angles(:,3) & ...
       dip_angles(:,3) <= dip_angles(:,4);

dip_angles = dip_angles(mask,:);
% dip_angles = [60 70 70 70 ]
vars = zeros(length(dip_angles),10);
for i = 1:size(dip_angles,1)
    
    fprintf("%d dip_angle  ",i);
    dip_angle = dip_angles(i,:)
    fprintf("\n");
slip_model_vs = load_fault_one_plane(fault_file,'dip',dip_angle,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'w_ratio',1.2,...
    'width',20e3,'len_top',2e3,'layers',5);

    [slip_model,rms1,model_roughness,var] = make_fault_from_insar1(slip_model_vs,slip_model_ds,iter_step,track, ...
                'ramp',ramp_choice,'segment_smooth_file',segment_file,'intersect_smooth_file',intersect_file,...
                'fault',fault_file,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'model_type','okada');
    vars(i,1:5) = dip_angle;
    vars(i,6:10) = var;
    fprintf("%d over!",i);
end
save("60-150free.mat","vars");
profile off
profile viewer
%% Step 3: apply quad-tree sampling to all detrended data 
make_insar_data(track,npt,region,Nmin,Nmax,'method','quadtree','fault',fault_file,'lonc',lonc,'latc',latc,'ref_lon',ref_lon);
%% Step 4: Build the fault geometry
% slip_model_vs = load_fault_one_plane(fault_file,'dip_change_id',dip_change_id,'dip',dip_angle,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,...
%     'width',20e3,'len_top',1e3,'layers',5);
slip_model_vs = load_fault_one_plane(fault_file,'dip',dip_angle,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'w_ratio',1.2,...
    'width',20e3,'len_top',3e3,'layers',5);

%% Step 5: inversion using first downsampled data
[slip_model,~,~] = make_fault_from_insar1(slip_model_vs,slip_model_ds,iter_step,track, ...
                'ramp',ramp_choice,'segment_smooth_file',segment_file,'intersect_smooth_file',intersect_file,...
                'fault',fault_file,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'model_type','okada');
% iint=iter_step;
show_slip_model(slip_model,'ref_lon',ref_lon,'lonc', lonc,'latc', latc,'axis_range',[50 150 -180 350 -20 0]);

% plot_insar_model_resampled(['./myanmar/D106/LOS/los_samp',num2str(iint),'.mat'],insar_model,'iter_step',iint,'fault',fault_file,'model_type','okada','misfit_range',30,'defo_max',120,'ref_lon',ref_lon,'lonc',lonc,'latc',latc);

%% Step 6: iterative sampling data using the model predictions (Wang and Fialko, GRL 2015) 
resamp_insar_data(slip_model,track,npt,Nmin,Nmax,data_types, iter_step2, 'fault', fault_file, 'dec',2,...
                    'lonc', lonc, 'latc', latc, 'ref_lon', ref_lon);

%% Step7: inversion using resampled data
iint=iter_step2;
[slip_model,rms1,model_roughness] = make_fault_from_insar1(slip_model_vs,slip_model_ds,iter_step2,track, ...
                     'segment_smooth_file',segment_file,'intersect_smooth_file',intersect_file,'fault',fault_file, ...
                     'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'ramp',ramp_choice);
show_slip_model(slip_model,'ref_lon',ref_lon,'lonc', lonc,'latc', latc,'axis_range',[50 150 -180 350 -20 0]);
%%
show_slip_model_2d(slip_model,'ref_lon',95,'lonc', 95.33,'latc', 19.61,'axis_range',[0 700 -30 0]);

plot_slip_vs_depth_arr(slip_model)

profile off
profile viewer