function [x, time] = BiCVaR(S, u, alpha, beta, sigma, mu, R, opts)

fprintf('***************** Bisection-Based CVaR ***************** \n');

time = 0; iter = 0;
alpha_l = alpha; alpha_u = 1; alpha_k = (alpha_l+alpha_u)/2;

while iter <= 100

    [x, Ctime] = CVaR(S, u, alpha_k, beta, sigma, mu, R, opts);
    time = time + Ctime;
    if risk_level(S, x, R) >= 1 - alpha
        alpha_l = alpha_k;
    else
        alpha_u = alpha_k;
    end

    if abs(alpha_u - alpha_l) > 1e-4
        alpha_k = (alpha_l+alpha_u)/2;
    else
        [x, Ctime] = CVaR(S, u, alpha_l, beta, sigma, mu, R, opts);
        time = time + Ctime;
        break;
    end
    iter = iter + 1;
    if time > 1800
        break;
    end
end
end