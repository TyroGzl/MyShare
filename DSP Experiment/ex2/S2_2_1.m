clear all
[data, Fs] = audioread('Hello.wav');
data_length = length(data);
data_rev = flip(data);
n = 0:data_length - 1;
w = 2 * n / (data_length);

[m, n] = size(data);

if n == 2
    data = data(:, 1);
end

data_dft = fft(data);
data_rev_dft = fft(data_rev);
figure
subplot(2, 2, 1);
plot(w, abs(data_dft));
title('原信号幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'Latex');
xlim([0, 0.2]);

subplot(2, 2, 2);
plot(w, abs(data_rev_dft));
title('翻转信号幅度谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'Latex');
xlim([0, 0.2]);

subplot(2, 2, 3);
plot(w, unwrap(angle(data_dft)));
title('原信号相位谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'Latex');
xlim([0, 0.2]);

subplot(2, 2, 4);
plot(w, unwrap(angle(data_rev_dft)));
title('翻转信号相位谱');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'Latex');
xlim([0, 0.2]);
