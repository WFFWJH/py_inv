function [ux, uy, uz] = calc_okada(HF, U, x, y, nu, delta, d, len, W, fault_type, strike, tp)
% Faster Okada displacement calculator
%
% HF         horizontal scaling factor
% U          slip
% x, y       observation points
% nu         Poisson ratio
% delta      dip angle
% d          top depth (positive down)
% len, W     fault length and width
% fault_type  1=strike, 2=dip, 3=tensile
% strike     strike angle
% tp         optional topo vector, same size as x/y

if nargin < 12 || isempty(tp)
    tp = 0;
end

% --- precompute trig values
cd = cos(delta);
sd = sin(delta);

% --- shift from top of fault to bottom of fault
d = d + W * sd;
x = x - W * cd * cos(strike);
y = y + W * cd * sin(strike);

% --- apply topography
d = d + tp;

% --- rotate to fault coordinates
strike2 = -strike + pi/2;
cs = cos(strike2);
sn = sin(strike2);

rotx =  x * cs + y * sn;
roty = -x * sn + y * cs;

% --- geometry constants
L = len * 0.5;
a = 1 - 2 * nu;
Const = -U / (2*pi);

% --- p and q are common to all 4 Okada terms
p = roty * cd + d * sd;
q = roty * sd - d * cd;

parvec = [a, delta, fault_type];

[f1a, f2a, f3a] = fBi_fast(rotx + L, p    , parvec, p, q);
[f1b, f2b, f3b] = fBi_fast(rotx + L, p - W, parvec, p, q);
[f1c, f2c, f3c] = fBi_fast(rotx - L, p    , parvec, p, q);
[f1d, f2d, f3d] = fBi_fast(rotx - L, p - W, parvec, p, q);

% --- displacement equations
uxj = Const * (f1a - f1b - f1c + f1d);
uyj = Const * (f2a - f2b - f2c + f2d);
uz  = Const * (f3a - f3b - f3c + f3d);

% --- rotate horizontals back
ux = HF * (-uyj * sn + uxj * cs);
uy = HF * ( uxj * sn + uyj * cs);

end


function [f1, f2, f3] = fBi_fast(sig, eta, parvec, p, q)
% Faster internal Okada kernel

a          = parvec(1);
delta      = parvec(2);
fault_type = parvec(3);

epsn = 1.0e-15;

cd = cos(delta);
sd = sin(delta);

% only used when cos(delta) ~= 0
if abs(cd) >= epsn
    td = sd / cd;
else
    td = 0;
end

cd2 = cd * cd;
sd2 = sd * sd;
cssd = cd * sd;

% --- common geometric terms
sig2 = sig .* sig;
eta2 = eta .* eta;
q2   = q   .* q;

R = sqrt(sig2 + eta2 + q2);
X = sqrt(sig2 + q2);

ytil = eta * cd + q * sd;
dtil = eta * sd - q * cd;

Rdtil = R + dtil;
Rsig  = R + sig;
Reta  = R + eta;
RX    = R + X;

lnRdtil = log(Rdtil);
lnReta  = log(Reta);

ORRsig = 1 ./ (R .* Rsig);
ORReta = 1 ./ (R .* Reta);
OReta  = 1 ./ Reta;

% --- fix singularities with logical indexing
mask = abs(Reta) < epsn;
if any(mask(:))
    lnReta(mask)  = -log(R(mask) - eta(mask));
    OReta(mask)   = 0;
    ORReta(mask)  = 0;
end

mask = abs(Rsig) < epsn;
if any(mask(:))
    ORRsig(mask) = 0;
end

% --- theta term
theta = zeros(size(sig), 'like', sig);
mask = abs(q) > epsn;
if any(mask(:))
    theta(mask) = atan((sig(mask) .* eta(mask)) ./ (q(mask) .* R(mask)));
end

% --- I terms
if abs(cd) < epsn
    % cos(delta) = 0 special case
    I5 = -a .* sig .* sd ./ Rdtil;
    I4 = -a .* q ./ Rdtil;
    I3 =  a/2 .* (eta ./ Rdtil + (ytil .* q) ./ (Rdtil.^2) - lnReta);
    I2 = -a .* lnReta - I3;
    I1 = -a/2 .* (sig .* q) ./ (Rdtil.^2);
else
    I5 = a * 2 ./ cd .* atan( (eta .* (X + q .* cd) + X .* RX .* sd) ./ (sig .* RX .* cd) );
    mask = abs(sig) < epsn;
    if any(mask(:))
        I5(mask) = 0;
    end

    I4 = a ./ cd .* (lnRdtil - sd .* lnReta);
    I3 = a .* (ytil ./ (cd .* Rdtil) - lnReta) + td .* I4;
    I2 = -a .* lnReta - I3;
    I1 = -a ./ cd .* sig ./ Rdtil - td .* I5;
end

% --- common reused products
sigqORReta = sig .* q .* ORReta;
yqORReta   = ytil .* q .* ORReta;
dqORReta   = dtil .* q .* ORReta;

yqORRsig   = ytil .* q .* ORRsig;
dqORRsig   = dtil .* q .* ORRsig;

qOReta = q .* OReta;

% --- fault-type specific formulas
switch fault_type
    case 1  % strike slip
        f1 = sigqORReta + theta + I1 .* sd;
        f2 = yqORReta   + (cd .* qOReta) + I2 .* sd;
        f3 = dqORReta   + (sd .* qOReta) + I4 .* sd;

    case 2  % dip slip
        f1 = q ./ R - I3 .* cssd;
        f2 = yqORRsig + cd .* theta - I1 .* cssd;
        f3 = dqORRsig + sd .* theta - I5 .* cssd;

    otherwise  % tensile
        f1 = q2 .* ORReta - I3 .* sd2;
        f2 = (-dqORRsig) - sd .* (sigqORReta - theta) - I1 .* sd2;
        f3 = yqORRsig + cd .* (sigqORReta - theta) - I5 .* sd2;
end

end