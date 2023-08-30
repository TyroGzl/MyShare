clear all
b = [2 -2];
a = [1 -1/3];
figure
zplane(b, a);
[r, p, k] = residuez(b, a)

n = 0:10;
h = -4 * (1/3).^n;
h(1) = -4 + 6;
figure
stem(n, h, 'filled', 'markersize', 3)
title('系统脉冲响应');
xlabel('$n$','Interpreter', 'Latex');
ylabel('$h[n]$','Interpreter', 'Latex');

y = 8 * (1/3).^n -6 * (1/2).^n;
figure
stem(n, y, 'filled', 'markersize', 3)
title('系统输出');
xlabel('$n$','Interpreter', 'Latex');
ylabel('$y[n]$','Interpreter', 'Latex');
