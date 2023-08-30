clear all
N1 = 100;
N2 = 100;
omega1 = pi / 16;
omega2 = 15 * pi / 16;
n1 = 0:N1 - 1;
n2 = 0:N2 - 1;
x1 = cos(omega1 * n1);
x2 = cos(omega2 * n1);
figure
subplot(2, 1, 1);
stem(n1, x1, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_1[n]$', 'Interpreter', 'latex');
title('$\omega_1=\pi/16$', 'Interpreter', 'latex');
subplot(2, 1, 2); stem(n2, x2, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
title('$\omega_2=15\pi/16$', 'Interpreter', 'latex');
