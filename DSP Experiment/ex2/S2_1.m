clear all
[data, Fs] = audioread('Hello.wav');
data_length = length(data);
n = 0:data_length - 1;
t = 16 * n / data_length;
w = 2 * n / data_length;
figure
plot(t, data); %绘制时域波形
title('时域波形');
xlabel('$t/s$', 'Interpreter', 'Latex');
ylabel('$x[n]$', 'Interpreter', 'Latex');
sound(data,Fs);%播放音频
[m, n] = size(data);

if n == 2
    data = data(:, 1);
end
%计算幅度谱以及相位谱
data_dft = fft(data);
figure
subplot(2, 1, 1);
plot(w, abs(data_dft));
title('幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'Latex');
xlim([-0.2, 2.2]);
subplot(2, 1, 2);
plot(w, unwrap(angle(data_dft)));
title('相位谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'Latex');
xlim([-0.2, 2.2]);
