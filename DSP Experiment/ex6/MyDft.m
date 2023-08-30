function [X] = MyDft(x, N)

    if length(x) < N
        x1 = zeros(1, N - length(x));
        x = [x, x1];
    end

    n = 0:N - 1;
    k = 0:N - 1;
    wn = exp(-j * 2 * pi / N);
    nk = n' .* k;
    wnk = wn.^nk;
    X = x * wnk;
end
