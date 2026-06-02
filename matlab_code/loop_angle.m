%% Geodetic Inversion Test
%11/15/2022, Xiaoyu Zou
%For Turkey earthquake inversion, try to start from Step 4
clear
close all
addpath("../OtherFunc/");
addpath("../detrend/")
addpath("../sign_mask/")
addpath("../sampling")
addpath("../Greens/") 
addpath("../geometry/")
addpath("../smooth/")
addpath("../inversion/")
addpath("../PLOT/")
% Step 0: set path and read in parameters & files
addpath(genpath('../geodetic_inversion-master'))
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
Nmin=para(7);
Nmax=para(8);
dip_change_id=para(9):para(10);%the array of fault ids that have dip angles not equal to 90 degrees
dip_angle=[para(11) para(12) para(13) para(14)];%the array of fault ids that have dip angles not equal to 90 degrees
iter_step=para(15);
iter_step2=para(16);
ramp_choice = "qu_ramp_7";

% dip_angle = [62 88 91 97 ]
% dip_angle = [65 65 80 80 ]

a = 60:10:150;

[A,B,C,D] = ndgrid(a,a,a,a);

dip_angles = [A(:), B(:), C(:), D(:)];
mask = dip_angles(:,1) < 90 & ...
       dip_angles(:,1) <= dip_angles(:,2) & ...
       dip_angles(:,2) <= dip_angles(:,3) & ...
       dip_angles(:,3) <= dip_angles(:,4);

dip_angles = dip_angles(mask,:);
% dip_angles = [60 70 70 70 ]
vars = zeros(length(dip_angles),9);
for i = 1:size(dip_angles,1)
    
    fprintf("%d dip_angle  ",i);
    dip_angle = dip_angles(i,:)
    fprintf("\n");
    slip_model_vs = load_fault_one_plane(fault_file,'dip_change_id',dip_change_id,'dip',dip_angle,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,...
        'width',35e3,'len_top',2e3,'layers',5);

    [slip_model,~,~,~] = make_fault_from_insar1(slip_model_vs,slip_model_ds,iter_step, ...
        'ramp',ramp_choice,'segment_smooth_file',segment_file,'intersect_smooth_file',intersect_file,...
        'fault',fault_file,'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'model_type','okada');

    resamp_insar_data(slip_model,data_list,Nmin, Nmax, iter_step2, 'fault', fault_file, 'dec',2,...
        'lonc', lonc, 'latc', latc, 'ref_lon', ref_lon);

    [slip_model,rms1,model_roughness,var] = make_fault_from_insar1(slip_model_vs,slip_model_ds,iter_step2, ...
        'segment_smooth_file',segment_file,'intersect_smooth_file',intersect_file,'fault',fault_file, ...
        'lonc',lonc,'latc',latc,'ref_lon',ref_lon,'ramp',ramp_choice);
    vars(i,1:4) = dip_angle;
    vars(i,5:9) = var;
    fprintf("%d over!",i);
end
save("60-150free.mat","vars");