clear all
N = 150;
n = 0:N - 1;
A1 = 1; omega1 = pi / 32; theta1 = 0;
A2 = 0.75; omega2 = 3 * pi / 32; theta2 = pi / 3;
A3 = 0.25; omega3 = 5 * pi / 32; theta3 = 2 * pi / 3;
x1 = A1 * cos(omega1 * n + theta1);
x2 = A2 * cos(omega2 * n + theta2);
x3 = A3 * cos(omega3 * n + theta3);
x = x1 + x2 + x3;
figure(1)
subplot(2, 1, 1);
stem(n, x1, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_1[n]$', 'Interpreter', 'latex');
title('$x_1[n]=cos(\frac{\pi}{32}n)$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(n, x2, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
title('$x_2[n]=0.75cos(\frac{3{\pi}}{32}n+\frac{\pi}{3})$', 'Interpreter', 'latex');
figure(2)
subplot(2, 1, 1);
stem(n, x3, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_3[n]$', 'Interpreter', 'latex');
title('$x_3[n]=0.25cos(\frac{5{\pi}}{32}n+\frac{2{\pi}}{3})$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(n, x, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
title('$x[n]=x_1[n]+x_2[n]+x_3[n]$', 'Interpreter', 'latex');
