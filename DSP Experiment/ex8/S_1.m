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
title('��ɢʱ���ĵ�ͼ�ź�');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
xlim([0, length(data) - 1]);
subplot(2, 1, 2);
plot(0:5 / (length(data) - 1):5, data);
title('����ʱ���ĵ�ͼ�ź�');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$x_c(t)$', 'Interpreter', 'latex');
xlim([0, 5]);
%��Ƶ������������
f = 50;
noise1 = 0.25 * cos(2 * pi * f / Fs * n + 1/4 * pi);
%���Ȱ�������������
noise2 = 0.12 * (rand(size(n)) - 0.5);
data_noise = data + noise1 + noise2;
figure
plot(t, data_noise);
title('�ܸ��ŵ��ĵ�ͼ�ź�');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$x_c(t)$', 'Interpreter', 'latex');
xlim([0, 5]);
Xk = fft(data_noise);
figure %����DFT������
subplot(2, 1, 1);
stem(n, abs(Xk), 'filled', 'markersize', 3);
title('�ܸ����ĵ�ͼ�ź�DFT������');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X[k]|$', 'Interpreter', 'latex');
xlim([0, length(Xk) - 1]);
subplot(2, 1, 2);
stem(n, 20 * log10(abs(Xk)), 'filled', 'markersize', 3);
title('�ܸ����ĵ�ͼ�ź�DFT�����׷ֱ���ʾ��ʽ');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X[k]|$', 'Interpreter', 'latex');
xlim([0, length(Xk) - 1]);
figure %����DTFT������
subplot(2, 1, 1);
plot(0:2 / (length(Xk) - 1):2, abs(Xk));
title('�ܸ����ĵ�ͼ�ź�DTFT������');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
xlim([0, 2]);
subplot(2, 1, 2);
plot(0:2 / (length(Xk) - 1):2, 20 * log10(abs(Xk)));
title('�ܸ����ĵ�ͼ�ź�DTFT�����׷ֱ���ʾ��ʽ');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X(e^{j\omega})|$', 'Interpreter', 'latex');
xlim([0, 2]);
figure %����CTFT������
subplot(2, 1, 1);
plot(0:Fs / (length(Xk) - 1):Fs, abs(Xk) / Fs);
title('�ܸ����ĵ�ͼ�ź�CTFT������');
xlabel('$f$', 'Interpreter', 'latex');
ylabel('$|X(f)|$', 'Interpreter', 'latex');
xlim([0, Fs / 2]);
subplot(2, 1, 2);
plot(0:Fs / (length(Xk) - 1):Fs, 20 * log10(abs(Xk) / Fs));
title('�ܸ����ĵ�ͼ�ź�CTFT�����׷ֱ���ʾ��ʽ');
xlabel('$f$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X(f)|$', 'Interpreter', 'latex');
xlim([0, Fs / 2]);
