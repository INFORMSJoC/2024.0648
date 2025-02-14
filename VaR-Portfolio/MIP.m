function [x, time] = MIP(S, upper, alpha, u, beta, sigma, mu, R, opts)

fprintf('***************** MIP ***************** \n')

%% parameter setting
[n, ~] = size(u); N = size(S,1);

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
    model.A = sparse([ones(1,n) sparse(1,N);...
        -S -speye(N)*upper; ...
        sparse(1,n) ones(1,N)]);
    model.lb = [zeros(1,n) -inf(1,N)];
    model.ub = [u' inf(1,N)];
    model.obj = [-mu zeros(1,N)];
    model.Q =[beta*sigma sparse(n,N); sparse(N,n) sparse(N,N)];
    model.modelsense = 'Min';
    model.rhs = [1 -R*ones(1,N) alpha*N];
    sign = ['='];
    for i=2:(N+2)
        sign=[sign,'<'];
    end
    model.sense = sign;

    model.vtype = [repmat('C',n,1); repmat('B',N,1)];

    %% Solve the problem using GUROBI
    params.outputflag = 1; % params.optimalitytol = 1e-9;
    params.timelimit = maxitime;
    result = gurobi(model, params);
    x = result.x(1:n);
    time = result.runtime;
else
    %% define variables
    x = sdpvar(n,1); y = binvar(N,1);

    %% Define constraints
    Constraints = [sum(y) <= alpha*N, ones(n,1)'*x == 1, zeros(n,1) <= x <= u];
    Constraints = [Constraints, R*ones(N,1)-S*x <= upper*y];

    %% Define an objective
    Objective = beta*x'*sigma*x-mu*x;

    %% solve by GUROBI solver
    ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 0, 'gurobi.timelimit', maxitime);
    sol = optimize(Constraints, Objective, ops);
    x = value(x);
    time = sol.solvertime;
end
end