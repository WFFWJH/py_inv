function plot_slip_vs_depth_arr(arr)
% arr columns:
% [fault, patch, depth, xp, yp, zp, lpatch, wpatch, strkp, dip, tp, slip1, slip2]

if size(arr,2) < 13
    error('arr must have 13 columns');
end

fault = arr(:,1);
depth = arr(:,3);
zp    = arr(:,6);
L     = arr(:,7);
W     = arr(:,8);
dip   = deg2rad(arr(:,10));   % 若本来就是弧度，删掉 deg2rad
slip  = hypot(arr(:,12), arr(:,13));

zcent = (zp - W .* sin(dip) / 2) / 1000;   % km
nDepth = max(depth);
segs   = unique(fault);

figure; hold on; grid on;

% ---- 全部数据 ----
cum_all = accumarray(depth, slip .* L, [nDepth,1], @sum, 0);
x_all = cum_all / max(cum_all);
y_all = accumarray(depth, zcent, [nDepth,1], @mean, NaN);
plot(x_all, y_all, 'k-s', 'LineWidth', 1.5, 'MarkerSize', 7);

legend_txt = {'all'};

% ---- 分段数据 ----
colors = lines(numel(segs));
for i = 1:numel(segs)
    k = segs(i);
    mask = fault == k;

    cum_seg = accumarray(depth(mask), slip(mask).*L(mask), [nDepth,1], @sum, 0);
    if max(cum_seg) > 0
        x_seg = cum_seg / max(cum_seg);
    else
        x_seg = cum_seg;
    end

    y_seg = accumarray(depth(mask), zcent(mask), [nDepth,1], @mean, NaN);
    plot(x_seg, y_seg, '-s', 'LineWidth', 1.2, 'MarkerSize', 6, 'Color', colors(i,:));

    legend_txt{end+1} = ['seg' num2str(k)];
end

xlabel('normalized cumulative slip');
ylabel('depth (km)');
xlim([0 1]);
% set(gca, 'YDir', 'reverse');
legend(legend_txt, 'Location', 'best');
end