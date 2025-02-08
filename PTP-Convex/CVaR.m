function [x,time] = CVaR(S, c, theta, alpha, opts)

%     fprintf('***************** CVaR ***************** \n');

%% parameter setting
[n, m] = size(c); N = size(S,1); % M = ceil((1-alpha)*N);
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
    model.A = sparse([sparse(1,n*m) 1 ones(1,N)/(alpha*N); kron(ones(1,m),speye(n)) sparse(n,1) sparse(n,N);...
        -kron(speye(m),ones(N,n)) -ones(N*m,1) -kron(ones(m,1),speye(N))]);
    model.lb = [zeros(1,n*m) -inf zeros(1,N)];
    model.ub = inf(1,n*m+1+N);
    model.obj = [c1 zeros(1,N+1)];
    model.modelsense = 'Min';
    model.rhs = [0 theta' -s];
    model.sense = '<';

    %% Solve the problem using GUROBI
    params.outputflag = 0;
    params.timelimit = maxitime;
    result = gurobi(model, params);
    x = reshape(result.x(1:n*m), [n, m]);
    time = result.runtime;

else
    %% define variables
    x = sdpvar(n,m); z = sdpvar(N,1); t = sdpvar(1,1);

    %% Define constraints
    Constraints = [t + sum(z)/(alpha*N) <= 0, z >= zeros(N,1), x >= zeros(n,m)];
    for i = 1:n
        Constraints = [Constraints, sum(x(i,:)) <= theta(i)];
    end
    for j = 1:m
        Constraints = [Constraints, S(:,j)-sum(x(:,j))*ones(N,1) - t*ones(N,1) <= z];
    end

    %% Define an objective
    Objective = trace(c'*x);

    %% solve by GUROBI solver
    ops = sdpsettings('solver', 'gurobi', 'verbose', 0, 'usex0',0, 'gurobi.timelimit', maxitime);
    sol = optimize(Constraints, Objective, ops);
    x = value(x);

end
end
