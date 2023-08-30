clear all;
close all;
x1 = [1 1 1 1];
N1 = 16;
N2 = 32;
N3 = 128;
x11 = [x1, zeros(1, N1 - length(x1))];
x12 = [x1, zeros(1, N2 - length(x1))];
x13 = [x1, zeros(1, N3 - length(x1))];
X1 = fft(x11);
X2 = fft(x12);
X3 = fft(x13);
figure
subplot(2, 1, 1);
stem(x11, 'filled', 'markersize', 3);
title('16点补零序列时域波形')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_1[n]$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(abs(X1), 'filled', 'markersize', 3);
title('16点补零序列幅频响应')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_1[k]|$', 'Interpreter', 'latex');
figure
subplot(2, 1, 1);
stem(x12, 'filled', 'markersize', 3);
title('32点补零序列时域波形')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(abs(X2), 'filled', 'markersize', 3);
title('32点补零序列幅频响应')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_2[k]|$', 'Interpreter', 'latex');
figure
subplot(2, 1, 1);
stem(x13, 'filled', 'markersize', 3);
title('128点补零序列时域波形')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_3[n]$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(abs(X3), 'filled', 'markersize', 3);
title('128点补零序列幅频响应')
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_3[k]|$', 'Interpreter', 'latex');
