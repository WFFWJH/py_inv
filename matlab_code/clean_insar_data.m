function clean_insar_data(tracks)

wavelength_C = 0.0554658*100;  % wavelength of C-band
wavelength_L = 0.242452*100;   % wavelength of L-band
scale = -4*pi;                 % for offsets data

for i = 1:numel(tracks)
    if strcmp(tracks{i}(end-2:end), 'insar')
        insar_file = 'unwrap_ll.grd';
mask_file = 'mask_txt';
mask_insar_phase(tracks{i},insar_file,mask_file,wavelength_C,'los_max',80,'detrend',1,'nomask',1);

    end

end