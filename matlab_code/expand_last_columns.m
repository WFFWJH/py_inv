function result = expand_last_columns(matrices, add_col, varargin)
% 功能：
% 将每个矩阵最后 add_col 列扩展为 n_classes * add_col
% 并按类别填入对应位置（类似 one-hot block 扩展）

% 输入：
% matrices   : cell，每个元素是一个矩阵
% add_col    : 要处理的最后几列
%
% 可选输入（二选一）：
% 'class_sizes', [n1,n2,...]  顺序分组
% 'class_map',   [1,1,2,...]  显式映射
%
% 输出：
% result     : cell，处理后的矩阵

% --------------------------------------------------
% 解析输入
% --------------------------------------------------
p = inputParser;
addParameter(p, 'class_sizes', []);
addParameter(p, 'class_map', []);
parse(p, varargin{:});

class_sizes = p.Results.class_sizes;
class_map   = p.Results.class_map;

n_m = numel(matrices);

% --------------------------------------------------
% 构造 class_map
% --------------------------------------------------
if ~isempty(class_sizes)
    if sum(class_sizes) ~= n_m
        error('sum(class_sizes) must equal number of matrices (%d).', n_m);
    end

    class_map = zeros(1, n_m);
    idx = 1;
    for cls = 1:numel(class_sizes)
        class_map(idx:idx+class_sizes(cls)-1) = cls;
        idx = idx + class_sizes(cls);
    end
    n_classes = numel(class_sizes);

elseif ~isempty(class_map)
    if numel(class_map) ~= n_m
        error('class_map length must equal number of matrices.');
    end
    n_classes = max(class_map);

else
    error('Provide either class_sizes or class_map');
end

% --------------------------------------------------
% 主处理
% --------------------------------------------------
result = cell(size(matrices));

for i = 1:n_m
    A = matrices{i};

    if size(A,2) < add_col
        error('Matrix %d has fewer columns than add_col.', i);
    end

    % 取最后几列
    lastCols = A(:, end-add_col+1:end);

    % 扩展块
    B = zeros(size(A,1), n_classes * add_col);
    cls = class_map(i);
    pos = (cls-1)*add_col + (1:add_col);
    B(:, pos) = lastCols;

    % 拼接
    ncolA = size(A,2);
    new_ncol = ncolA + (n_classes-1)*add_col;

    C = zeros(size(A,1), new_ncol);
    C(:,1:ncolA-add_col) = A(:,1:ncolA-add_col);
    C(:,ncolA-add_col+1:end) = B;

    result{i} = C;
end

end