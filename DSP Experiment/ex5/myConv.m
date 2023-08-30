function [y, ny] = myConv(x, nx, h, nh)
    [h1, nh1] = seqFlip(h, nh);
    y = [];
    ny = [];

    for m = min(nx):min(nx) + (length(x) + length(h) - 1) - 1
        [h2, nh2] = seqShift(h1, nh1, m);
        [y1, ny1] = seqMult(x, nx, h2, nh2);
        yn = sum(y1);
        y = [y, yn];
        ny = [ny, m];
    end

end

function [y, n] = seqAdd(x1, n1, x2, n2)
    n = min(min(n1), min(n2)):max(max(n1), max(n2));
    y1 = zeros(1, length(n));
    y2 = y1;
    y1(find((n >= min(n1)) & (n <= max(n1)) == 1)) = x1;
    y2(find((n >= min(n2)) & (n <= max(n2)) == 1)) = x2;
    y = y1 + y2;
end

function [y, n] = seqMult(x1, n1, x2, n2)
    n = min(min(n1), min(n2)):max(max(n1), max(n2));
    y1 = zeros(1, length(n));
    y2 = y1;
    y1(find((n >= min(n1)) & (n <= max(n1)) == 1)) = x1;
    y2(find((n >= min(n2)) & (n <= max(n2)) == 1)) = x2;
    y = y1 .* y2;
end

function [y, ny] = seqShift(x, nx, k)
    y = x;
    ny = nx + k;
end

function [y, ny] = seqFlip(x, nx)
    y = flip(x);
    ny = -flip(nx);
end
