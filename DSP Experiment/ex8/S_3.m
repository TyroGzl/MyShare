clear;
close all;
fid = fopen('Ecg360.txt');
data = fscanf(fid, "%f");
data = data';
Fs = 360;
n = 0:length(data) - 1;
%工频干扰序列生成
f = 50;
noise1 = 0.25 * cos(2 * pi * f / Fs * n + 1/4 * pi);
%均匀白噪声序列生成
noise2 = 0.12 * (rand(size(n)) - 0.5);
data_noise = data + noise1 + noise2;

wc = 0.2 * pi;
dw = 0.08 * pi;
A = 40;
beta = 0.5842 * (A - 21)^0.4 + 0.07886 * (A - 21);
N = ceil((A - 7.95) / (2.285 * dw) + 1);
wn = kaiser(N, beta);
[W, ww] = freqz(wn, 1);
alpha = (N - 1) / 2;

for n = 0:N - 1
    hdn(n + 1) = sin(wc * (n - alpha)) ./ (pi * (n - alpha));

    if n == alpha
        hdn(n + 1) = wc / pi;
    end

end

hn = hdn .* wn';

n = 0:360 * 5 + N - 1 - 1;
data_line = conv(data_noise, hn);
figure;
subplot(2, 1, 1);
plot(n, [data_noise, zeros(1, N - 1)]);
title('补零后受干扰心电图信号');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
xlim([0, length(n) - 1])
subplot(2, 1, 2);
plot(n, data_line);
title('线性卷积滤波后信号');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_{f1}[n]$', 'Interpreter', 'latex');
xlim([0, length(n) - 1])

data_circle = myCircleConv(data_noise, hn, 360 * 5 + N - 1);
figure;
subplot(2, 1, 1);
plot(n, [data_noise, zeros(1, N - 1)]);
title('补零后受干扰心电图信号');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
xlim([0, length(n) - 1])
subplot(2, 1, 2);
plot(n, data_circle);
title('线性卷积滤波后信号');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_{f2}[n]$', 'Interpreter', 'latex');
xlim([0, length(n) - 1])

D1 = fft(data_noise, length(n));
D2 = fft(data_circle);
figure
subplot(2, 1, 1);
plot(n, 20 * log10(abs(D1)));
title('受干扰心电图信号幅度谱');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X[k]|$', 'Interpreter', 'latex');
xlim([0, length(n) / 2]);
subplot(2, 1, 2);
plot(n, 20 * log10(abs(D2)));
title('滤波后信号幅度谱');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X_{f}[k]|$', 'Interpreter', 'latex');
xlim([0, length(n) / 2]);
