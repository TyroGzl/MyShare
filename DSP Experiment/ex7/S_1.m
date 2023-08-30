clear all;
close all;
Fs = 360;
fid = fopen('Ecginf.txt');
data = fscanf(fid, '%f', inf);
figure
subplot(2, 1, 1)
stem(0:length(data) - 1, data, 'filled', 'markersize', 1);
title('�����ĵ�ͼ�ź�')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
xlim([0, length(data) - 1]);
subplot(2, 1, 2)
plot(0:60 / (length(data) - 1):60, data);
title('ģ���ĵ�ͼ�ź�')
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$x_c(t)$', 'Interpreter', 'latex');

Xk = fft(data);
figure
subplot(2, 2, 1);
stem(0:(length(Xk) - 1), abs(Xk), 'filled', 'markersize', 3);
title('DFT������')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X[k]|$', 'Interpreter', 'latex');
xlim([0, length(Xk) - 1]);
subplot(2, 2, 2);
stem(0:(length(Xk) - 1), 20 * log10(abs(Xk)), 'filled', 'markersize', 3);
title('DFT�����׷ֱ���ʾ��ʽ')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X[k]|(dB)$', 'Interpreter', 'latex');
xlim([0, length(Xk) - 1]);
subplot(2, 2, 3);
plot(0:2 / (length(Xk) - 1):2, abs(Xk));
title('DTFT������')
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 2, 4);
plot(0:2 / (length(Xk) - 1):2, 20 * log10(abs(Xk)));
title('DTFT�����׷ֱ���ʾ��ʽ')
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X(e^{j\omega})|(dB)$', 'Interpreter', 'latex');
figure
subplot(2, 1, 1);
plot(0:Fs / (length(Xk) - 1):Fs, abs(Xk));
title('CTFT������')
xlabel('$f$', 'Interpreter', 'latex');
ylabel('$|X(f)|$', 'Interpreter', 'latex');
subplot(2, 1, 2);
plot(0:Fs / (length(Xk) - 1):Fs, 20 * log10(abs(Xk)));
title('CTFT�����׷ֱ���ʾ��ʽ')
xlabel('$f$', 'Interpreter', 'latex');
ylabel('$20log_{10}|X(f)|(dB)$', 'Interpreter', 'latex');
