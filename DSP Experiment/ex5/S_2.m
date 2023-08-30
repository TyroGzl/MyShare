close all;
N = 12;
M = 20;
L = 28;
x1 = ones(1, N);
x2 = ones(1, M);
y = myCircleConv(x1, x2, L);
n = 0:L - 1;
figure
subplot(3, 1, 1)
stem(n, [x1, zeros(1, L - N)], 'filled', 'markersize', 3);
title('$x_1[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_1[n]$', 'Interpreter', 'latex');
subplot(3, 1, 2)
stem(n, [x2, zeros(1, L - M)], 'filled', 'markersize', 3);
title('$x_2[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
subplot(3, 1, 3)
stem(n, y, 'filled', 'markersize', 3);
title('圆周卷积结果');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_c[n]$', 'Interpreter', 'latex');
