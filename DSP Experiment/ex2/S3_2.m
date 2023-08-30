clear all
samples = [1, 12 * 32 * 1000];
[data, Fs] = audioread('intro.mp3', samples);

data_length = length(data);
n = 0:data_length - 1;
t = 12 * n / data_length;
w = 2 * n / data_length;

for i = 1:5
    d(i) = Fs * (i - 1) / 10;
end

delay1 = zeros(d(1), 1);
delay2 = zeros(d(2), 1);
delay3 = zeros(d(3), 1);
delay4 = zeros(d(4), 1);
delay5 = zeros(d(5), 1);

data_d1 = [delay1; data];
data_d1 = data_d1(1:data_length,1);
data_d2 = [delay2; data / 2];
data_d2 = data_d2(1:data_length,1);
data_d3 = [delay3; data / 4];
data_d3 = data_d3(1:data_length,1);
data_d4 = [delay4; data / 8];
data_d4 = data_d4(1:data_length,1);
data_d5 = [delay5; data / 16];
data_d5 = data_d5(1:data_length,1);

data_sum = data_d1 + data_d2 + data_d3 + data_d4 + data_d5;

figure
subplot(2,1,1);
plot(t, data); %绘制时域波形
title('原始信号时域波形');
xlabel('$t/s$', 'Interpreter', 'Latex');
ylabel('$x[n]$', 'Interpreter', 'Latex');
subplot(2,1,2);
plot(t, data_sum); %绘制时域波形
title('和弦信号时域波形');
xlabel('$t/s$', 'Interpreter', 'Latex');
ylabel('$x[n]$', 'Interpreter', 'Latex');

data_sum_dft = fft(data_sum);

figure
subplot(2, 1, 1);
plot(w, abs(data_sum_dft));
title('幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'Latex');
xlim([-0.2, 2.2]);
subplot(2, 1, 2);
plot(w, 20 * log(abs(data_sum_dft)));
title('分贝形式');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|/dB$', 'Interpreter', 'Latex');
xlim([-0.2, 2.2]);
