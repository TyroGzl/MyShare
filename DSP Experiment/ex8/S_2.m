close all;
clear;
wc = 0.2 * pi;
dw = 0.08 * pi;
A = 40;
beta = 0.5842 * (A - 21)^0.4 + 0.07886 * (A - 21);
N = ceil((A - 7.95) / (2.285 * dw) + 1);
wn = kaiser(N, beta);
[W, ww] = freqz(wn, 1);
alpha = (N - 1) / 2;
figure
subplot(2, 1, 1)
stem(wn, 'filled', 'markersize', 3);
title('Kaiser窗口序列')
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$w[n]$', 'Interpreter', 'latex');
xlim([0, length(wn)]);
subplot(2, 1, 2)
plot(ww / pi, 20 * log10(abs(W)));
title('Kaiser窗口序列频谱')
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$W(e^{j\omega})$', 'Interpreter', 'latex');
xlim([0, 1]);

for n = 0:N - 1
    hdn(n + 1) = sin(wc * (n - alpha)) ./ (pi * (n - alpha));

    if n == alpha
        hdn(n + 1) = wc / pi;
    end

end

n = 0:N - 1;
hn = hdn .* wn';
figure
subplot(2, 1, 1);
stem(n, hdn, 'filled', 'markersize', 3);
title('理想数字滤波器单位脉冲响应');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h_d[n]$', 'Interpreter', 'latex');
subplot(2, 1, 2);
stem(hn, 'filled', 'markersize', 3);
title('FIR数字滤波器单位脉冲响应');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
figure
zplane(hn)
[H, wh] = freqz(hn, 1);
[gd, wh1] = grpdelay(hn, 1);
figure
subplot(2, 1, 1);
plot(wh / pi, abs(H));
title('FIR数字滤波器幅度响应');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$|H(e^{j\omega})|$', 'Interpreter', 'latex');
subplot(2, 1, 2);
plot(wh / pi, 20 * log10(abs(H)));
title('FIR数字滤波器幅度响应分贝表示形式');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$20log_{10}|H(e^{j\omega})|$', 'Interpreter', 'latex');
figure
subplot(3, 1, 1);
plot(wh / pi, angle(H));
title('FIR数字滤波器相位响应');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$\angle{H(e^{j\omega})}$', 'Interpreter', 'latex');
subplot(3, 1, 2);
plot(wh / pi, unwrap(angle(H)));
title('FIR数字滤波器相位响应（解缠绕）');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H(e^{j\omega})]$', 'Interpreter', 'latex');
subplot(3, 1, 3);
plot(wh1 / pi, gd);
title('FIR数字滤波器群延迟');
xlabel('$\omega(*\pi)$', 'Interpreter', 'latex');
ylabel('$grd[H(e^{j\omega})]$', 'Interpreter', 'latex');
