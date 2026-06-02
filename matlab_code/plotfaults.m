file1 = 'fault';
file2 = 'fault_trace1.txt';

% 读取数据（强烈推荐用 load）
seg1 = load(file1);
seg2 = load(file2);

figure; hold on;

%% ===== 画第一个文件（蓝色）=====
for i = 1:size(seg1,1)
    h1=plot([seg1(i,1), seg1(i,3)], ...
         [seg1(i,2), seg1(i,4)], ...
         'b-', 'LineWidth', 1.5);
end

%% ===== 画第二个文件（红色）=====
for i = 1:size(seg2,1)
    h2=plot([seg2(i,1), seg2(i,3)], ...
         [seg2(i,2), seg2(i,4)], ...
         'r-', 'LineWidth', 1.5);
end

%% 图形设置
axis equal
xlabel('Longitude')
ylabel('Latitude')
title('Comparison of Two Fault Segment Files')
legend([h1, h2], {'File 1','File 2'});
