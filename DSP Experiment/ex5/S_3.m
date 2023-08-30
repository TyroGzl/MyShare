close all
N = 12;
M = 20;
nx = 0:N - 1;
x = ones(1, N);
nh = 0:M - 1;
h = 0.98.^nh;
L = [M, 2 * N, N + M - 1, 2 * M];
n1 = 0:L(1) - 1;
n2 = 0:L(2) - 1;
n3 = 0:L(3) - 1;
n4 = 0:L(4) - 1;
yc1 = myCircleConv(x, h, L(1));
yc2 = myCircleConv(x, h, L(2));
yc3 = myCircleConv(x, h, L(3));
yc4 = myCircleConv(x, h, L(4));

figure
subplot(3, 1, 1)
stem(n1, [x, zeros(1, L(1) - N)], 'filled', 'markersize', 3);
title('$x[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
subplot(3, 1, 2)
stem(n1, [h, zeros(1, L(1) - M)], 'filled', 'markersize', 3);
title('$h[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
subplot(3, 1, 3)
stem(n1, yc1, 'filled', 'markersize', 3);
title('20点圆周卷积结果');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_{c1}[n]$', 'Interpreter', 'latex');

figure
subplot(3, 1, 1)
stem(n2, [x, zeros(1, L(2) - N)], 'filled', 'markersize', 3);
title('$x[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
subplot(3, 1, 2)
stem(n2, [h, zeros(1, L(2) - M)], 'filled', 'markersize', 3);
title('$h[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
subplot(3, 1, 3)
stem(n2, yc2, 'filled', 'markersize', 3);
title('24点圆周卷积结果');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_{c2}[n]$', 'Interpreter', 'latex');

figure
subplot(3, 1, 1)
stem(n3, [x, zeros(1, L(3) - N)], 'filled', 'markersize', 3);
title('$x[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
subplot(3, 1, 2)
stem(n3, [h, zeros(1, L(3) - M)], 'filled', 'markersize', 3);
title('$x[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
subplot(3, 1, 3)
stem(n3, yc3, 'filled', 'markersize', 3);
title('31点圆周卷积结果');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_{c3}[n]$', 'Interpreter', 'latex');

figure
subplot(3, 1, 1)
stem(n4, [x, zeros(1, L(4) - N)], 'filled', 'markersize', 3);
title('$x[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
subplot(3, 1, 2)
stem(n4, [h, zeros(1, L(4) - M)], 'filled', 'markersize', 3);
title('$h[n]$', 'Interpreter', 'latex');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
subplot(3, 1, 3)
stem(n4, yc4, 'filled', 'markersize', 3);
title('40点圆周卷积结果');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_{c4}[n]$', 'Interpreter', 'latex');

y3 = myCircleConv(x, h, 31);
y4 = myConv(x, nx, h, nh);
figure
subplot(2, 2, 1)
stem(nx, x, 'filled', 'markersize', 3);
title('x[n]');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
subplot(2, 2, 2)
stem(nh, h, 'filled', 'markersize', 3);
title('h[n]');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
subplot(2, 2, 3)
stem(y3, 'filled', 'markersize', 3);
title('圆周卷积实现');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_3[n]$', 'Interpreter', 'latex');
subplot(2, 2, 4)
stem(y4, 'filled', 'markersize', 3);
title('自写函数实现');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_4[n]$', 'Interpreter', 'latex');
