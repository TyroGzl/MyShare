clear;
close all;
fid = fopen('Ecg360.txt');
data = fscanf(fid, "%f");
data = data';
Fs = 360;
n = 0:length(data) - 1;
t = 0:5 / (length(data) - 1):5;
figure
subplot(2, 1, 1);
stem(data, 'filled', 'markersize', 3);
title('离散时间心电图信号');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
xlim([0, length(data) - 1]);
subplot(2, 1, 2);
plot(0:5 / (length(data) - 1):5, data);
title('连续时间心电图信号');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$x_c(t)$', 'Interpreter', 'latex');
xlim([0, 5]);
%工频干扰序列生成
f = 50;
noise1 = 0.25 * cos(2 * pi * f / Fs * n + 1/4 * pi);
%均匀白噪声序列生成
noise2 = 0.12 * (rand(size(n)) - 0.5);
data_noise = data + noise1 + noise2;
figure
plot(t, data_noise);
title('受干扰的心电图信号');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$x_c(t)$', 'Interpreter', 'latex');
xlim([0, 5]);
Xk = fft(data_noise);
figure %绘制DFT幅度谱
subplot(2, 1, 1);
stem(n, abs(Xk), 'filled', 'markersize', 3);
title('受干扰心电图信号DFT幅度谱');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X[k]|$', 'Interpreter', 'latex');
xlim([0, length(Xk) - 1]);
subplot(2, 1, 2);
stem(n, 20 * log10(abs(Xk)), 'filled', 'markersize', 3);
title('受干扰心电图信号DFT幅度谱分贝表示形式');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X[k]|$', 'Interpreter', 'latex');
xlim([0, length(Xk) - 1]);
figure %绘制DTFT幅度谱
subplot(2, 1, 1);
plot(0:2 / (length(Xk) - 1):2, abs(Xk));
title('受干扰心电图信号DTFT幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
xlim([0, 2]);
subplot(2, 1, 2);
plot(0:2 / (length(Xk) - 1):2, 20 * log10(abs(Xk)));
title('受干扰心电图信号DTFT幅度谱分贝表示形式');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X(e^{j\omega})|$', 'Interpreter', 'latex');
xlim([0, 2]);
figure %绘制CTFT幅度谱
subplot(2, 1, 1);
plot(0:Fs / (length(Xk) - 1):Fs, abs(Xk) / Fs);
title('受干扰心电图信号CTFT幅度谱');
xlabel('$f$', 'Interpreter', 'latex');
ylabel('$|X(f)|$', 'Interpreter', 'latex');
xlim([0, Fs / 2]);
subplot(2, 1, 2);
plot(0:Fs / (length(Xk) - 1):Fs, 20 * log10(abs(Xk) / Fs));
title('受干扰心电图信号CTFT幅度谱分贝表示形式');
xlabel('$f$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X(f)|$', 'Interpreter', 'latex');
xlim([0, Fs / 2]);
