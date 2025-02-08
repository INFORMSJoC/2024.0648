function [x_value, time] = MIP(S, U, upper, alpha, opts)

fprintf('***************** MIP ***************** \n')

%% parameter setting
[N, n, m] = size(S);

if isfield(opts,'solver')
    solver = opts.solver;
else
    solver = 'gurobi';
end
if isfield(opts,'maxitime')
    maxitime = opts.maxitime;
else
    maxitime = 1800;
end
if isfield(opts, 'tol')
    tol = opts.tol;
else
    tol = 1e-4;
end
%% CVX
cvxtime = tic;
cvx_begin quiet
    variable x(n,1)
    variable y(N,1) binary

    Constraints = [sum(y) <= alpha*N, x >= zeros(n,1)];
    for j = 1:m
        Constraints = [Constraints, (S(:,:,j).^2)*(x.^2) - U <= upper*y];
    end

    cvx_solver gurobi
    cvx_solver_settings('OptimalityTol', tol)
    cvx_solver_settings('TimeLimit', maxitime)
    Objective = -sum(x);
    minimize(Objective)
    subject to
        Constraints
cvx_end

time = toc(cvxtime);
x_value = x;
end
