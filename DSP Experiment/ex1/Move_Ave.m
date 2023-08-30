function [y] = Move_Ave(m, x)
    [rows, cols] = size(x);

    for i = 1:cols
        sum = 0;

        if i >= (m + 1) / 2 && i <= (cols - (m - 1) / 2)

            for j = i - (m - 1) / 2:i + (m - 1) / 2
                sum = x(j) + sum;
            end

            y(i) = sum / m;
        else
            y(i) = x(i);
        end

    end

end
