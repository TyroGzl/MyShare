function [y] = myCircleConv(x, h, L)
    M = length(x);
    N = length(h);
    x1 = zeros(1, L - M);
    h1 = zeros(1, L - N);
    x = [x, x1];
    h = [h, h1];
    X = fft(x, L);
    H = fft(h, L);
    Y = X .* H;
    y = ifft(Y);
    y = y(:, 1:L);
end
