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
[hz, wz] = freqz(bz, az);
[hn, n] = impz(bz, az);
figure
stem(hn, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
figure
zplane(bz, az);
figure
subplot(2, 1, 1);
plot(wz / pi, abs(hz));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 1, 2);
plot(wz / pi, 20 * log10(abs(hz)));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$20log_{10}|H(e^{j\omega})|$', 'Interpreter', 'latex');
