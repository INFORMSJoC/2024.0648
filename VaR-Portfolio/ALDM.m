function [x,iter,time] = ALDM(beta, S, alpha, R, u, sigma, mu, opts)

% solve MIP reformulation:
% min f(x)
% s.t. y_i = g(x,xi^i), i=1,...,N,
%      y_i >= z_i*d_i, i=1,...,N,
%      sum^N(p_i*z_i) <= alpha,
%      z \in {0,1}^N, x \in X.

fprintf('***************** ALDM ***************** \n')
%% parameter setting
[n, ~] = size(u); N = size(S,1); MAXIter = 1e3;

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

time = 0;
lambda = zeros(N,1);
kappa = 1; rho = 1;
H = 2*beta*sigma+10e-5*rand(1)*eye(n);

%% generate the initial point by solving min{f(x): x \in X}
if isfield(opts,'x0')
    x = opts.x0;
else
    model.A = sparse(ones(1,n));
    model.lb = zeros(1,n);
    model.ub = u';
    model.Q =sparse(beta*sigma);
    model.obj = -mu;
    model.modelsense = 'Min';
    model.rhs = 1;

    model.sense = '=';
    %% Solve the problem using GUROBI
    params.outputflag = 0;
    params.timelimit = maxitime;
    result = gurobi(model, params);
    x = result.x;

end
%% compute lower bound of each g(x,\xi^i) for i = 1,\dots,N by d_i = min{g(x,\xi^i): x \in X}
if isfield(opts, 'd')
    d = opts.d;
else
    d = -100*ones(N,1);
end

start = tic;
for iter = 1:MAXIter
    %% update (y,z) when fixing (lambda,x)
    [~, y, ~] = ALDM_update_y(S, x, R, lambda, rho, d, alpha);

    %% update x when fixing (lambda,y,z)
    x = ALDM_update_x_g(beta, S, y, u, x, sigma, mu, R, rho, lambda, H, opts);

    %% stopping criteria
    g = S*x-R*ones(N,1);
    prob = risk_level(S, x, R);
    r = norm(y-g,'fro')^2;

    %% update dual variable lambda
    lambda = lambda + kappa*rho*(y-g);
    rho = min(rho * 1.05, 1e3);
    %% report iterate information
    time = toc(start);
    if mod(iter, 40) == 0
        fprintf('Iternum: %d, iter gap: %1.2e, prob: %.4f\n', iter, r, prob);
    end

    if (r <= tol) || time > maxitime % (r <= 1e-4) && prob >= 1-alpha
        fprintf('Iternum: %d, iter gap: %1.2e, prob: %.4f\n', iter, r, prob);
        break;
    end
end
end