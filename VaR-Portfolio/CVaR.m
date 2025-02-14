function [x, time] = CVaR(S, u, alpha, beta, sigma, mu, R, opts)

% fprintf('***************** CVaR ***************** \n');

%% parameter setting
[n, ~] = size(u);
N = size(S,1);

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
if strcmp(solver,'gurobi') == 1
    %% Model parameters in GUROBI
    model.A = sparse([ones(1,n) sparse(1,N+1);...
        sparse(1,n) 1 ones(1,N)/(alpha*N); ...
        -S -ones(N,1) -speye(N)]);
    model.lb = [zeros(1,n) -inf zeros(1,N)];
    model.ub = [u' inf(1,N+1)];
    model.Q =[beta*sigma sparse(n,N+1); sparse(N+1,n) sparse(N+1,N+1)];
    model.obj = [-mu zeros(1,N+1)];
    model.modelsense = 'Min';
    model.rhs = [1 0 -R*ones(1,N)];
    sign = ['='];
    for i=2:(N+2)
        sign=[sign,'<'];
    end
    model.sense = sign;

    %% Solve the problem using GUROBI
    params.outputflag = 0;
    params.timelimit = maxitime;
    result = gurobi(model, params);
    x = result.x(1:n);
    time = result.runtime;
else
    %% define variables
    x = sdpvar(n,1); z = sdpvar(N,1); t = sdpvar(1,1);

    %% Define constraints
    Constraints = [ones(n,1)'*x == 1, zeros(n,1) <= x <= u, ...
        t + ones(N,1)'*z/(alpha*N) <= 0, z >= zeros(N,1)];
    Constraints = [Constraints, R*ones(N,1)-S*x - t*ones(N,1) <= z];

    %% Define an objective
    Objective = beta*x'*sigma*x-mu*x;

    %% solve by GUROBI solver
    ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0',0, 'gurobi.timelimit', maxitime);
    sol = optimize(Constraints, Objective, ops);
    time = sol.solvertime;
    x = value(x);
end