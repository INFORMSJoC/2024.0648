function [x, iter] = BiCVaR(S, U, alpha, opts)

fprintf('***************** Bisection-Based CVaR ***************** \n');

iter = 0; [~, ~, m] = size(S);
alpha_l = alpha; alpha_u = 1; alpha_k = (alpha_l+alpha_u)/2;

tic;
while iter <= 1e2

    [x,~] = CVaR(S, U, alpha_k, opts);
    if risk_level(S, x, m) >= 1 - alpha
        alpha_l = alpha_k;
    else
        alpha_u = alpha_k;
    end

    if abs(alpha_u - alpha_l) > 1e-2
        alpha_k = (alpha_l+alpha_u)/2;
    else
        [x,~] = CVaR(S, U, alpha_l, opts);
        break;
    end
    iter = iter + 1;
    if toc > 1800
        break;
    end
end
end