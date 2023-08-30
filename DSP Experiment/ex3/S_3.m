b = [1 -1.8 -1.44 0.64];
a = [1 -1.6485 1.03882 -0.288];
figure
zplane(b, a)
[H, w] = freqz(b, a, 'whole');

figure
plot(w / pi, abs(H));
title('幅频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$|H(e^{j\omega})|$', 'Interpreter', 'Latex');
figure
plot(w / pi, angle(H))
title('相频特性');
xlabel('$\omega(*\pi)$', 'Interpreter', 'Latex');
ylabel('$\angle{H(e^{j\omega})}$', 'Interpreter', 'Latex');
