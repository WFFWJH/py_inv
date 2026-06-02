function resamp_insar_data(slip_model_in, track,npt,Nmin,Nmax,data_types,iter_step,varargin)
% resample the insar data with model predictions 
% method in Kang Wang and Yuri Fialko, GRL 2015
% written by Zeyu Jin on Sept. 2019
   set(0,'defaultAxesFontSize',15);
   
   % data_list defines each Interferogram track with such a format
   % the number means the number of data points to be resampled
   % ASC99    1200
   % DES106   1500
   
   % slip_model_in is the previous inverted model
   
   %% DEFINE THE DEFAULT VALUES
   iint=iter_step;

   Nlook = 1;   
   %% read varargin values and assembly
   if ~isempty(varargin)
       for CC = 1:floor(length(varargin)/2)
           try
               switch lower(varargin{CC*2-1})
                   case 'lonc'
                       lon_eq = varargin{CC*2};
                   case 'latc'
                       lat_eq = varargin{CC*2};
                   case 'fault'
                       fault_file = varargin{CC*2};
                   case 'dec'
                       Nlook = varargin{CC*2};
                   
                   case 'ref_lon'
                       ref_lon = varargin{CC*2};
               end
           catch
               error('Unrecognized Keyword\n');
           end
       end
   end
   

   [xo,yo] = ll2xy(lon_eq,lat_eq,ref_lon);
   
   % % data covariance matrix
   % varigram=load('insar_varigram.mat');
   % sigma=varigram.sigma;
   % L=varigram.L;
   
   % iterative sample the data using model predictions
   for k=1:numel(track)
       this_track=track{k};
       data_type = data_types{k};
       disp(['working on ',this_track, 'type: ', data_type]);
       this_npt=npt(k);
       
       [~,~,demin]=grdread2([this_track,'/','dem_low.grd']);
       [~,~,losin]=grdread2([this_track,'/','los_ll_low','.grd']);   % in the unit of cm
       [~,~,ze]=grdread2([this_track,'/','look_e_low','.grd']);
       [~,~,zn]=grdread2([this_track,'/','look_n_low','.grd']);
       [x1,y1,zu]=grdread2([this_track,'/','look_u_low','.grd']);
       
       % multi-look to reduce the computation time
       if Nlook > 1
          [~,~,deml] = multi_look(x1,y1,demin,Nlook,Nlook);
          [~,~,losl] = multi_look(x1,y1,losin,Nlook,Nlook);
          [~,~,zel] = multi_look(x1,y1,ze,Nlook,Nlook);
          [~,~,znl] = multi_look(x1,y1,zn,Nlook,Nlook);
          [lon1,lat1,zul] = multi_look(x1,y1,zu,Nlook,Nlook);
       else
           lon1 = x1;
           lat1 = y1;
           deml = demin;
           losl = losin;
           zel = ze;
           znl = zn;
           zul = zu;
       end
       
       [xm1,ym1] = meshgrid(lon1,lat1);
       [xutm,yutm] = ll2xy(xm1(:),ym1(:),ref_lon);
       xin = xutm - xo;
       yin = yutm - yo;
       xin = reshape(xin,size(xm1));
       yin = reshape(yin,size(ym1));
       
       if strcmp(data_type,'insar')
           disp('insar');
          los_model = slip2insar_okada(xin,yin,losl,zel,znl,zul,slip_model_in);   % fix the bug using multi-looked looking angles
       elseif strcmp(data_type,'rng')
           disp('rng');
          los_model = slip2insar_okada(xin,yin,losl,zel,znl,zul,slip_model_in);   % fix the bug using multi-looked looking angles
          
       elseif strcmp(data_type,'azi')
           disp('azi');
          los_model = slip2AZO_okada(xin,yin,losl,zel,znl,slip_model_in);     % add module to compute AZO data
          
       end
       
       % resample the data using slip model
       % a bit similar to EM algorithm to make it diverge (E-step??  M-step?? Currently I don't know...)

       [lon_model,lat_model,~,~,rms_out,xx1,xx2,yy1,yy2]=make_insar_downsample(lon1,lat1,los_model,this_npt,Nmin(k),Nmax(k),'mean'); % same with downsample

       [~,~,zout]=make_look_downsample(lon1,lat1,losl,lon_model,lat_model,xx1,xx2,yy1,yy2);
       [~,~,dem_out]=make_look_downsample(lon1,lat1,deml,lon_model,lat_model,xx1,xx2,yy1,yy2);
       [~,~,ve]=make_look_downsample(lon1,lat1,zel,lon_model,lat_model,xx1,xx2,yy1,yy2);
       [~,~,vn]=make_look_downsample(lon1,lat1,znl,lon_model,lat_model,xx1,xx2,yy1,yy2);
       [lon_pt,lat_pt,vz]=make_look_downsample(lon1,lat1,zul,lon_model,lat_model,xx1,xx2,yy1,yy2);
       
       % convert lon/lat to x/y
       [xutm,yutm]=ll2xy(lon_pt,lat_pt,ref_lon);
       xpt=xutm-xo;
       ypt=yutm-yo;
       
       indx_good=~isnan(zout);
       xpt=xpt(indx_good);
       ypt=ypt(indx_good);
       zout=zout(indx_good);
       dem_out=dem_out(indx_good);
       rms_out=rms_out(indx_good);
       ve=ve(indx_good);
       vn=vn(indx_good);
       vz=vz(indx_good);
       xx1=xx1(indx_good);
       yy1=yy1(indx_good);
       xx2=xx2(indx_good);
       yy2=yy2(indx_good);
           
       sampled_insar_data = double([xpt,ypt,zout,ve,vn,vz]);
       save([this_track,'/los_samp',num2str(iint),'.mat'],'sampled_insar_data','rms_out','dem_out');
       % [hf,~,~]=plot_insar_sample_new(x1,y1,losin,zout,xx1,xx2,yy1,yy2,'fault',fault_file);
       % set(hf,'PaperPositionMode','auto');
       % set(hf,'visible','off');
%       saveas(hf,[this_track,'/','los_samp',num2str(iint)],'epsc');
   end
   
end