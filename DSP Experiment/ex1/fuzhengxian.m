clear all
r = 0.9;
omega = pi / 8;
N = 50;
n = 0:N - 1;
x = r.^n .* exp(j * omega * n);
figure
subplot(2, 1, 1);
stem(n, real(x), 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$Re(x[n])$', 'Interpreter', 'latex');
title('余弦分量');
subplot(2, 1, 2);
stem(n, imag(x), 'filled', 'markersize', 3);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$Im(x[n])$', 'Interpreter', 'latex');
title('正弦分量');
