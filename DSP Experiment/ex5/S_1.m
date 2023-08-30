x = [3, 11, 7, 0, -1, 4, 2];
nx = [0, 1, 2, 3, 4, 5, 6];
h = [2, 3, 0, -5, 2, 1];
nh = [0, 1, 2, 3, 4, 5];
[y1, ny1] = myConv(x, nx, h, nh)
y2 = conv(x, h)
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
stem(ny1, y1, 'filled', 'markersize', 3);
title('自己编写的线性卷积实现');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_1[n]$', 'Interpreter', 'latex');
subplot(2, 2, 4)
stem(y2, 'filled', 'markersize', 3);
title('conv实现');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y_2[n]$', 'Interpreter', 'latex');
