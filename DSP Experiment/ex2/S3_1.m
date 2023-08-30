clear all
samples = [1, 12 * 32 * 1000];
[data, Fs] = audioread('intro.mp3', samples);

sound(data,Fs);

data_length = length(data);
n = 0:data_length - 1;
t = 12 * n / data_length;
w = 2 * n / data_length;

figure
plot(t, data); %绘制时域波形
title('时域波形');
xlabel('$t/s$', 'Interpreter', 'Latex');
ylabel('$x[n]$', 'Interpreter', 'Latex');

[m, n] = size(data);

if n == 2
    data = data(:, 1);
end

data_dft = fft(data);

figure
subplot(2, 1, 1);
plot(w, abs(data_dft));
title('幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'Latex');
xlim([-0.2, 2.2]);
subplot(2, 1, 2);
plot(w, 20 * log(abs(data_dft)));
title('分贝形式');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|/dB$', 'Interpreter', 'Latex');
xlim([-0.2, 2.2]);
