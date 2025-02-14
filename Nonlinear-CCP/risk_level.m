function ratio = risk_level(S, x, m)

N = size(S, 1);
count = 0;

for i = 1:N
    for j = 1:m
        if 100 - (S(i,:,j).^2)*(x.^2) < -1e-1
            count = count + 1;
            break;
        end
    end
end

ratio = 1 - count/N;

end