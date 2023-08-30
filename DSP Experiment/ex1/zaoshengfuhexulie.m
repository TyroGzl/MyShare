clear all
A = 1;
omega = pi / 16;
theta = 0;
N = 150;
n = 0:N - 1;
s = A * cos(omega * n + theta);
rand_z = rand(size(s));
z = (rand_z - 0.5) * 0.6;
x = s + z;
figure(1)
subplot(3, 1, 1);
stem(n, s, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$s[n]$', 'Interpreter', 'latex');
title('正弦序列');
subplot(3, 1, 2);
stem(n, z, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$z[n]$', 'Interpreter', 'latex');
title('噪声序列');
subplot(3, 1, 3);
stem(n, x, 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
title('添加噪声后的正弦序列');
