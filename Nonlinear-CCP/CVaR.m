function [x_value, time] = CVaR(S, U, alpha, opts)

%     fprintf('***************** CVaR ***************** \n');

%% parameter setting
[N, n, m] = size(S);
% M = ceil((1-alpha)*N);

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

%% CVX
cvxtime = tic;
cvx_begin quiet
    variables x(n) t z(N)
    Constraints = [t + sum(z)/(alpha*N) <= 0, z >= zeros(N,1), x >= zeros(n,1)];
    for j = 1:m
        Constraints = [Constraints, (S(:,:,j).^2)*(x.^2) - U*ones(N,1) - t*ones(N,1) <= z];
    end
    
    Objective = -sum(x);
%     cvx_solver sdpt3
    cvx_solver gurobi
    cvx_solver_settings('OptimalityTol', 1e-4)
    cvx_solver_settings('TimeLimit', maxitime)
    minimize(Objective)
    subject to
        Constraints
cvx_end

time = toc(cvxtime);
x_value = x;
end
