clear all
close all
b = [1 10/3 1];
a = [1];
figure
zplane(b, a);
title('系统函数的零极点图')

%最小相位系统
b_min = [1 2/3 1/9];
a_min = [1];
figure
zplane(b_min, a_min);
title('最小相位零极点图')
[H_min, w1_min] = freqz(b_min, a_min, 'whole');
[gd_min, w2_ap] = grpdelay(b_min, a_min, 1000, 'whole');
figure
subplot(3, 1, 1);
plot(w1_min / pi, abs(H_min));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H_{min}(z)|$', 'Interpreter', 'latex');
title('最小相位系统幅频响应')
subplot(3, 1, 2);
plot(w1_min / pi, angle(H_min));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H_{min}(z)}$', 'Interpreter', 'latex');
title('最小相位系统相位响应')
subplot(3, 1, 3);
plot(w2_ap / pi, gd_min);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H_{min}(z)]$', 'Interpreter', 'latex');
title('最小相位系统群延迟响应')

%全通系统
b_ap = [1/3 1];
a_ap = [1 1/3];
figure
zplane(b_ap, a_ap);
title('全通系统零极点图')
[H_ap, w1_ap] = freqz(b_ap, a_ap, 'whole');
[gd_ap, w2_ap] = grpdelay(b_ap, a_ap, 1000, 'whole');
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

%级联
b1 = 3 * conv(b_min, b_ap);
a1 = conv(a_min, a_ap);
figure
impz(b1, a1);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
title('级联后单位脉冲响应');
