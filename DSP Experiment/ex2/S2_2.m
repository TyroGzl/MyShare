clear all
[data, Fs] = audioread('Hello.wav');
data_rev = flip(data);
data_length = length(data);
n = 0:data_length - 1;
t = 16 * n / data_length;
w = 2 * n / data_length;
figure
subplot(2,1,1);
plot(t, data); %绘制时域波形
title('时域波形');
xlabel('$t/s$', 'Interpreter', 'Latex');
ylabel('$x[n]$', 'Interpreter', 'Latex');
subplot(2,1,2);
plot(t, data_rev);
title('翻转后时域波形');
xlabel('$t/s$', 'Interpreter', 'Latex');
ylabel('$x[n]$', 'Interpreter', 'Latex');

[m, n] = size(data);

if n == 2
    data = data(:, 1);
end

data_dft = fft(data_rev);
figure
subplot(2, 1, 1);
plot(w, abs(data_dft));
title('幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'Latex');
xlim([-0.2,2.2]);
subplot(2, 1, 2);
plot(w,unwrap(angle(data_dft)));
title('相位谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'Latex');
xlim([-0.2,2.2]);
