function [x, time, iter] = ALDM(S, c, a, theta, alpha, opts)

% solve MIP reformulation:
% min f(x)
% s.t. y_i = g(x,xi^i), i=1,...,N,
%      y_i >= z_i*d_i, i=1,...,N,
%      sum^N(p_i*z_i) <= alpha,
%      z \in {0,1}^N, x \in X.

fprintf('***************** ALDM ***************** \n');

%% parameter setting
[n, m] = size(c); N = size(S,1); maxiter = 1e3;

if isfield(opts,'solver')
    solver = opts.solver;
else
    solver = 'gorubi';
end
if isfield(opts,'maxitime')
    maxitime = opts.maxitime;
else
    maxitime = 1800;
end
if isfield(opts, 'tol')
    tol = opts.tol;
else
    tol = 1e-6;
end
time = 0; kappa = 0.01; rho = 1;
lambda = zeros(N,m);

%% generate the initial point by solving min{f(x): x \in X}
if isfield(opts,'x0')
    x = opts.x0;
else
    x = sdpvar(n,m);
    Constraints = [x >= zeros(n,m)];
    for i = 1:n
        Constraints = [Constraints, sum(x(i,:)) <= theta(i)];
    end
    Objective = trace(c'*x);
    ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0',0, 'gurobi.timelimit', maxitime);
    sol = optimize(Constraints, Objective, ops);
    time = time + sol.solvertime;
    x = value(x);
end
d = -S;

tic;
for iter = 1:maxiter

    %% update (y,z) when fixing (lambda,x)
    [~, y, ~] = ALDM_update_y(S, c, x, lambda, rho, d, alpha);

    %% update x when fixing (lambda,y,z)
    x = ALDM_update_x(S, c, a, theta, y, lambda, rho, x, opts);

    %% stopping criteria
    g = ones(N,1)*sum(x) - S;
    x(x<1e-6) = 0;
    prob = risk_level(S, x, m);
    r = norm(y-g,'fro')^2;

    %% update dual variable lambda
    lambda = lambda + kappa*rho*(y-g);
    rho = min(rho * 1.05, 1e3);

    %% report iterate information
    time = toc;
    fval = trace(c'*x)+trace(a'*x.^2);
    if mod(iter, 20) == 0
        fprintf('Iternum: %d, iter gap: %1.2e, fval: %.4f, prob: %.4f\n', iter, r, fval, prob);
    end
    if (r <= 1e-6) || time > 1800
        fprintf('Iternum: %d, iter gap: %1.2e, prob: %.4f\n', iter, r, prob);
        break;
    end

end
end
