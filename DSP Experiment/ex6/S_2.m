clear all;
close all;
x2 = [3, 2, 1, 3, -2, -1, 2, 4];
X3 = fft(x2, 512);
figure
stem(x2, 'filled', 'markersize', 3);
title('原始数字序列')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
figure
subplot(2, 1, 1);
plot(0:2 / (length(X3) - 1):2, abs(X3));
title('幅频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 1, 2);
plot(0:2 / (length(X3) - 1):2, angle(X3));
title('相频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'latex');

%第二部分
X4 = fft(x2, 16);
X5 = fft(x2, 64);
%离散频谱特性
figure
subplot(2, 2, 1);
stem(0:(length(X4) - 1), abs(X4), 'filled', 'markersize', 3);
title('16点DFT幅频特性');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_{4}[k]|$', 'Interpreter', 'latex');
subplot(2, 2, 2);
stem(0:(length(X4) - 1), angle(X4), 'filled', 'markersize', 3);
title('16点DFT相频特性');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$\angle{X_{4}[k]}$', 'Interpreter', 'latex');
subplot(2, 2, 3);
stem(0:(length(X5) - 1), abs(X5), 'filled', 'markersize', 3);
title('64点DFT幅频特性');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_{5}[k]|$', 'Interpreter', 'latex');
subplot(2, 2, 4);
stem(0:(length(X5) - 1), angle(X5), 'filled', 'markersize', 3);
title('64点DFT相频特性');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$\angle{X_{5}[k]}$', 'Interpreter', 'latex');
%连续频谱特性
figure
subplot(2, 2, 1);
plot(0:2 / (length(X4) - 1):2, abs(X4));
title('16点幅频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 2, 2);
plot(0:2 / (length(X4) - 1):2, angle(X4));
title('16点相频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'latex');
subplot(2, 2, 3);
plot(0:2 / (length(X5) - 1):2, abs(X5));
title('64点幅频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 2, 4);
plot(0:2 / (length(X5) - 1):2, angle(X5));
title('64点相频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'latex');
