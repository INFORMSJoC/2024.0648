function ratio = risk_level(train_samples, x, R)

N = size(train_samples, 1);
count = 0;

for i = 1:N
    if train_samples(i,:)*x-R < -1e-4
        count = count + 1;
        %             break;
    end
end

ratio = 1 - count/N;

end