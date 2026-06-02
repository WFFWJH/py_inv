function zout = slip2insar_okada(XIN, YIN, zin, look_e, look_n, look_z, slip_model_in, varargin)

% ---------- optional ----------
nu = 0.25;
if ~isempty(varargin)
    for ii = 1:2:numel(varargin)
        if strcmpi(varargin{ii}, 'poisson')
            nu = varargin{ii+1};
        end
    end
end

d2r = pi/180;
HF = 1;

% ---------- model ----------
xp    = slip_model_in(:,4);
yp    = slip_model_in(:,5);
zp    = slip_model_in(:,6);
lp    = slip_model_in(:,7);
wp    = slip_model_in(:,8);
strkp = slip_model_in(:,9);
dip0  = slip_model_in(:,10);
s1    = slip_model_in(:,12);
s2    = slip_model_in(:,13);

Npatch = numel(xp);

theta = (90.0 - strkp) * d2r;
[dx, dy] = xy2XY(lp/2, 0, -theta);

xxo = xp + dx;
yyo = yp + dy;
zzo = zp;

delta  = dip0 * d2r;
strike = strkp * d2r;
d      = -zzo;

% ---------- observation ----------
good = ~isnan(zin);

xinsar = XIN(good);
yinsar = YIN(good);
ve     = look_e(good);
vn     = look_n(good);
vz     = look_z(good);

ngood = numel(xinsar);

% ---------- ⭐ 像元分块 ----------
pixelBlockSize = 5e5;   % ⭐关键参数（50万~100万）
np = ceil(ngood / pixelBlockSize);

zout_good = zeros(ngood,1);

% ---------- patch block ----------
blockSize = 3;
nb = ceil(Npatch / blockSize);

for ip = 1:np

    % ===== 当前像元块 =====
    i1 = (ip-1)*pixelBlockSize + 1;
    i2 = min(ip*pixelBlockSize, ngood);

    x_sub = xinsar(i1:i2);
    y_sub = yinsar(i1:i2);
    ve_sub = ve(i1:i2);
    vn_sub = vn(i1:i2);
    vz_sub = vz(i1:i2);

    z_sub = zeros(i2-i1+1,1);

    % ===== patch 分块 =====
    for ib = 1:nb

        k1 = (ib-1)*blockSize + 1;
        k2 = min(ib*blockSize, Npatch);

        tmp_cell = cell(k2-k1+1,1);

        parfor kk = 1:(k2-k1+1)

            k = k1 + kk - 1;

            % --- 局部变量 ---
            x0 = xxo(k);
            y0 = yyo(k);
            delta_k = delta(k);
            d_k = d(k);
            lp_k = lp(k);
            wp_k = wp(k);
            strike_k = strike(k);
            U1 = s1(k);
            U2 = s2(k);

            % --- 计算 ---
            xpt = x_sub - x0;
            ypt = y_sub - y0;

            [ue1, un1, uz1] = calc_okada_two(HF, U1,U2, xpt, ypt, nu, delta_k, d_k, lp_k, wp_k, 1, strike_k);

            tmp_cell{kk} = ue1 .* ve_sub + un1 .* vn_sub + uz1 .* vz_sub;

        end
        % --- 累加 ---
        for kk = 1:numel(tmp_cell)
            z_sub = z_sub + tmp_cell{kk};
        end

    end
    % ===== 写回 =====
    zout_good(i1:i2) = z_sub;

end

% ---------- output ----------
zout = NaN(size(zin));
zout(good) = zout_good;

end