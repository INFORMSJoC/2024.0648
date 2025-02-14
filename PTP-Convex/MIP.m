function [x,time] = MIP(S, c, theta, upper, alpha, opts)

fprintf('***************** MIP ***************** \n')

%% parameter setting
[n, m] = size(c); N = size(S,1);
s = reshape(S,[1,N*m]); c1 = reshape(c,[1,n*m]);

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
    model.A = sparse([kron(ones(1,m),speye(n)) sparse(n,N);...
        -kron(speye(m),ones(N,n)) -kron(ones(m,1),speye(N))*upper; sparse(1,n*m) ones(1,N)]);
    model.lb = [zeros(1,n*m) zeros(1,N)];
    model.ub = [inf(1,n*m) ones(1,N)];
    model.obj = [c1 zeros(1,N)];
    model.modelsense = 'Min';
    model.rhs = [theta' -s alpha*N];
    model.sense = '<';
    model.vtype = [repmat('C',n*m,1); repmat('B',N,1)];

    %% Solve the problem using GUROBI
    params.outputflag = 1; params.optimalitytol = 1e-9;
    params.timelimit = maxitime;
    result = gurobi(model, params);
    x = reshape(result.x(1:n*m), [n, m]);
    time = result.runtime;
else
    %% define variables
    x = sdpvar(n,m); y = binvar(N,1);

    %% Define constraints
    Constraints = [sum(y) <= alpha*N, x >= zeros(n,m)];
    for i = 1:n
        Constraints = [Constraints, sum(x(i,:)) <= theta(i)];
    end
    for j = 1:m
        Constraints = [Constraints, S(:,j)-sum(x(:,j))*ones(N,1) <= upper*y];
    end

    %% Define an objective
    Objective = trace(c'*x);

    %% solve by GUROBI solver
    ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 0, 'gurobi.timelimit', maxitime);
    optimize(Constraints, Objective, ops);
    x = value(x);

end
end
