function [x, iter] = BiCVaR(S, c, theta, alpha, opts)

fprintf('***************** Bisection-Based CVaR ***************** \n');

iter = 0; [~, m] = size(c);
alpha_l = alpha; alpha_u = 1; alpha_k = (alpha_l+alpha_u)/2;

tic;
while iter <= 100
    [x,~] = CVaR(S, c, theta, alpha_k, opts);
    if risk_level(S, x, m) >= 1 - alpha
        alpha_l = alpha_k;
    else
        alpha_u = alpha_k;
    end

    if abs(alpha_u - alpha_l) > 1e-4
        alpha_k = (alpha_l+alpha_u)/2;
    else
        [x,~] = CVaR(S, c, theta, alpha_l, opts);
        break;
    end
    iter = iter + 1;
    if toc > 1800
        break;
    end
end
end