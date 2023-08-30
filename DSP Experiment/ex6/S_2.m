clear all;
close all;
x2 = [3, 2, 1, 3, -2, -1, 2, 4];
X3 = fft(x2, 512);
figure
stem(x2, 'filled', 'markersize', 3);
title('ԭʼ��������')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x_2[n]$', 'Interpreter', 'latex');
figure
subplot(2, 1, 1);
plot(0:2 / (length(X3) - 1):2, abs(X3));
title('��Ƶ����');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 1, 2);
plot(0:2 / (length(X3) - 1):2, angle(X3));
title('��Ƶ����');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'latex');

%�ڶ�����
X4 = fft(x2, 16);
X5 = fft(x2, 64);
%��ɢƵ������
figure
subplot(2, 2, 1);
stem(0:(length(X4) - 1), abs(X4), 'filled', 'markersize', 3);
title('16��DFT��Ƶ����');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_{4}[k]|$', 'Interpreter', 'latex');
subplot(2, 2, 2);
stem(0:(length(X4) - 1), angle(X4), 'filled', 'markersize', 3);
title('16��DFT��Ƶ����');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$\angle{X_{4}[k]}$', 'Interpreter', 'latex');
subplot(2, 2, 3);
stem(0:(length(X5) - 1), abs(X5), 'filled', 'markersize', 3);
title('64��DFT��Ƶ����');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$|X_{5}[k]|$', 'Interpreter', 'latex');
subplot(2, 2, 4);
stem(0:(length(X5) - 1), angle(X5), 'filled', 'markersize', 3);
title('64��DFT��Ƶ����');
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$\angle{X_{5}[k]}$', 'Interpreter', 'latex');
%����Ƶ������
figure
subplot(2, 2, 1);
plot(0:2 / (length(X4) - 1):2, abs(X4));
title('16���Ƶ����');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 2, 2);
plot(0:2 / (length(X4) - 1):2, angle(X4));
title('16����Ƶ����');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'latex');
subplot(2, 2, 3);
plot(0:2 / (length(X5) - 1):2, abs(X5));
title('64���Ƶ����');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|X(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 2, 4);
plot(0:2 / (length(X5) - 1):2, angle(X5));
title('64����Ƶ����');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{X(e^{j\omega})}$', 'Interpreter', 'latex');
