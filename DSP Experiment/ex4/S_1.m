clear all
close all
b = [1 10/3 1];
a = [1];
figure
zplane(b, a);
title('ϵͳ�������㼫��ͼ')

%��С��λϵͳ
b_min = [1 2/3 1/9];
a_min = [1];
figure
zplane(b_min, a_min);
title('��С��λ�㼫��ͼ')
[H_min, w1_min] = freqz(b_min, a_min, 'whole');
[gd_min, w2_ap] = grpdelay(b_min, a_min, 1000, 'whole');
figure
subplot(3, 1, 1);
plot(w1_min / pi, abs(H_min));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H_{min}(z)|$', 'Interpreter', 'latex');
title('��С��λϵͳ��Ƶ��Ӧ')
subplot(3, 1, 2);
plot(w1_min / pi, angle(H_min));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H_{min}(z)}$', 'Interpreter', 'latex');
title('��С��λϵͳ��λ��Ӧ')
subplot(3, 1, 3);
plot(w2_ap / pi, gd_min);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H_{min}(z)]$', 'Interpreter', 'latex');
title('��С��λϵͳȺ�ӳ���Ӧ')

%ȫͨϵͳ
b_ap = [1/3 1];
a_ap = [1 1/3];
figure
zplane(b_ap, a_ap);
title('ȫͨϵͳ�㼫��ͼ')
[H_ap, w1_ap] = freqz(b_ap, a_ap, 'whole');
[gd_ap, w2_ap] = grpdelay(b_ap, a_ap, 1000, 'whole');
figure
subplot(3, 1, 1);
plot(w1_ap / pi, abs(H_ap));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H_{ap}(z)|$', 'Interpreter', 'latex');
title('ȫͨϵͳ��Ƶ��Ӧ')
subplot(3, 1, 2);
plot(w1_ap / pi, unwrap(angle(H_ap)));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H_{ap}(z)}$', 'Interpreter', 'latex');
title('ȫͨϵͳ��λ��Ӧ')
subplot(3, 1, 3);
plot(w2_ap / pi, gd_ap);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H_{ap}(z)]$', 'Interpreter', 'latex');
title('ȫͨϵͳȺ�ӳ���Ӧ')

%����
b1 = 3 * conv(b_min, b_ap);
a1 = conv(a_min, a_ap);
figure
impz(b1, a1);
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
title('������λ������Ӧ');
