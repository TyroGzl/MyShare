clear all
close all
alpha = 0.5;
%ԭϵͳ
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
title('ԭϵͳ��Ƶ��Ӧ')
subplot(3, 1, 2);
plot(w1 / pi, angle(H1));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H(z)}$', 'Interpreter', 'latex');
title('ԭϵͳ��λ��Ӧ')
subplot(3, 1, 3);
plot(w2 / pi, gd1);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H(z)]$', 'Interpreter', 'latex');
title('ԭϵͳȺ�ӳ���Ӧ')

%ȫͨϵͳ
b_ap=[0.0445 -0.132 0.3703 -0.5103 1];
a_ap = [1 -0.5103 0.3703 -0.1320 0.0445];
[H_ap,w1_ap] = freqz(b_ap,a_ap);
[gd_ap,w2_ap] = grpdelay(b_ap,a_ap);
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
figure
zplane(b_ap,a_ap)
title('ȫͨϵͳ�㼫��ֲ�ͼ')

%��λ����
b = b0 * conv(b1,b_ap);
a = conv(a1,a_ap);
figure
zplane(b,a)
title('��λ������㼫��ͼ')
[H,w11] = freqz(b,a);
[gd,w22] = grpdelay(b,a);
figure
subplot(3, 1, 1);
plot(w11 / pi, abs(H));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H(z)|$', 'Interpreter', 'latex');
title('��λ������Ƶ��Ӧ')
subplot(3, 1, 2);
plot(w11 / pi, unwrap(angle(H)));
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H(z)}$', 'Interpreter', 'latex');
title('��λ�������λ��Ӧ')
subplot(3, 1, 3);
plot(w22 / pi, gd);
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H(z)]$', 'Interpreter', 'latex');
title('��λ�����Ⱥ�ӳ���Ӧ')

