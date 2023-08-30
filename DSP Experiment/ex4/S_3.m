clear all
close all
Fs = 44100;
data = importdata('newaudio_44100_4s.mat');
N = length(data);
n = 1:N;
figure
plot(n / N * 4, data)
xlabel('$t/s$', 'Interpreter', 'latex');
ylabel('$x(t)$', 'Interpreter', 'latex');
title('原音频信号时域波形');
sound(data, Fs);

len_d = 0.5/4 * N;
delay = zeros(len_d, 1);
data1 = [delay; data];
data1 = data1(1:N, 1);
y = data + data1;
figure
plot(n / N * 4, y)
xlabel('$t/s$', 'Interpreter', 'latex');
ylabel('$y(t)$', 'Interpreter', 'latex');
title('添加回声后音频信号时域波形');
sound(y, Fs);

z = zeros(N, 1);

for index = len_d +1:N
    z(index) = y(index) - z(index - len_d);
end

figure
plot(n / N * 4, z)
xlabel('$t/s$', 'Interpreter', 'latex');
ylabel('$z(t)$', 'Interpreter', 'latex');
title('消除回声后音频信号时域波形');
sound(z, Fs);
