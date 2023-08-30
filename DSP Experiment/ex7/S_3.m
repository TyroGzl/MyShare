clear all;
close all;
Fs = 360;
wp = 0.2 * pi;
ws = 0.3 * pi;
dp = 1/100;
ds = 1/1000;
ap = -20 * log10(1 - dp);
as = -20 * log10(ds);
Omega_p = (2 * Fs) * tan(wp / 2);
Omega_s = (2 * Fs) * tan(ws / 2);
[N, wc] = buttord(Omega_p, Omega_s, ap, as, 's');
[b, a] = butter(N, wc, 's');
[bz, az] = bilinear(b, a, Fs);
[hn, n] = impz(bz, az, 21600);
fid = fopen('Ecginf.txt');
data = fscanf(fid, '%f', inf);

Xk = fft(data);
Hk = fft(hn);
Xfk = Xk .* Hk;
dataf = ifft(Xfk);
figure
subplot(2, 2, 1);
plot(abs(Xk));
title('滤波前幅度谱')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X[k]|$', 'Interpreter', 'latex');
xlim([0, 21600]);
subplot(2, 2, 2);
plot(20 * log10(abs(Xk)));
title('滤波前幅度谱分贝表示形式')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X[k]|(dB)$', 'Interpreter', 'latex');
xlim([0, 21600]);
subplot(2, 2, 3);
plot(abs(Xfk));
title('滤波后幅度谱')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X[k]|$', 'Interpreter', 'latex');
xlim([0, 21600])
subplot(2, 2, 4);
plot(20 * log10(abs(Xfk)));
title('滤波后幅度谱分贝表示形式')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X[k]|(dB)$', 'Interpreter', 'latex');
xlim([0, 21600])

figure
subplot(2, 1, 1);
plot(data);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
xlim([0, 21600])
subplot(2, 1, 2);
plot(dataf);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_f[n]$', 'Interpreter', 'latex');
xlim([0, 21600])
