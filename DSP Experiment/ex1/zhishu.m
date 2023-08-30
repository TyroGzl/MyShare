clear all
N1 = 30;
N2 = 30;
n1 = 0:N1 - 1;
n2 = 0:N2 - 1;
r1 = 0.9;
r2 = -0.9;
x1 = r1.^n1;
x2 = r2.^n2;
figure
subplot(2, 1, 1);
stem(n1, x1, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_1[n]$', 'Interpreter', 'latex');
title('$r_1=0.9$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(n2, x2, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
title('$r_2=-0.9$', 'Interpreter', 'latex');
