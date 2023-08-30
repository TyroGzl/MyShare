clear all
close all
alpha = 0.5;
%原系统
b0 = (1-alpha)/2;
b1 = [1 1];
a1 = [1 -alpha];
[H1,w1] = freqz(b0*b1,a1);
[gd1,w2] = grpdelay(b0*b1,a1);
figure
subplot(3, 1, 1);
plot(w1 / pi, abs(H1));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H(z)|$', 'Interpreter', 'latex');
title('原系统幅频响应')
subplot(3, 1, 2);
plot(w1 / pi, angle(H1));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H(z)}$', 'Interpreter', 'latex');
title('原系统相位响应')
subplot(3, 1, 3);
plot(w2 / pi, gd1);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H(z)]$', 'Interpreter', 'latex');
title('原系统群延迟响应')

%全通系统
b_ap=[0.0445 -0.132 0.3703 -0.5103 1];
a_ap = [1 -0.5103 0.3703 -0.1320 0.0445];
[H_ap,w1_ap] = freqz(b_ap,a_ap);
[gd_ap,w2_ap] = grpdelay(b_ap,a_ap);
figure
subplot(3, 1, 1);
plot(w1_ap / pi, abs(H_ap));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H_{ap}(z)|$', 'Interpreter', 'latex');
title('全通系统幅频响应')
subplot(3, 1, 2);
plot(w1_ap / pi, unwrap(angle(H_ap)));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H_{ap}(z)}$', 'Interpreter', 'latex');
title('全通系统相位响应')
subplot(3, 1, 3);
plot(w2_ap / pi, gd_ap);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H_{ap}(z)]$', 'Interpreter', 'latex');
title('全通系统群延迟响应')
figure
zplane(b_ap,a_ap)
title('全通系统零极点分布图')

%相位均衡
b = b0 * conv(b1,b_ap);
a = conv(a1,a_ap);
figure
zplane(b,a)
title('相位均衡后零极点图')
[H,w11] = freqz(b,a);
[gd,w22] = grpdelay(b,a);
figure
subplot(3, 1, 1);
plot(w11 / pi, abs(H));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H(z)|$', 'Interpreter', 'latex');
title('相位均衡后幅频响应')
subplot(3, 1, 2);
plot(w11 / pi, unwrap(angle(H)));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H(z)}$', 'Interpreter', 'latex');
title('相位均衡后相位响应')
subplot(3, 1, 3);
plot(w22 / pi, gd);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H(z)]$', 'Interpreter', 'latex');
title('相位均衡后群延迟响应')

