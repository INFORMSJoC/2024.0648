function x = ALDM_update_x(S, c, a, theta, y, lambda, rho, x0, opts)

% subproblem (6):
% given lambda; fix y and z.
% min L(x,\bary,\barlambda) = f(x) + sum^N(lambda^T(\bary-g(x,xi))) + rho/2*sum^N||\bary-g(x,xi)||^2,
% s.t. x \in X.

%% parameter setting
[n, m] = size(c); N = size(S,1);
c1 = reshape(c,[1,n*m]); a1 = reshape(a,[1,n*m]);
H = 1e-5*rand(1)*eye(n*m);
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
    model.A = sparse(kron(ones(1,m),speye(n)));
    model.obj = (c1+2*a1.*x0(:)')-ones(1,N)*lambda*kron(speye(m),ones(1,n))-rho*(ones(1,N)*y+ones(1,N)*S)*kron(speye(m),ones(1,n))-x0(:)'*H;
    model.Q = sparse((N*rho/2)*(kron(speye(m),ones(1,n)))'*(kron(speye(m),ones(1,n)))+0.5*H);
    model.lb = zeros(1,n*m);
    model.ub = inf(1,n*m);
    model.modelsense = 'Min';
    model.rhs = theta';
    model.sense = '<';

    %% Solve the problem using GUROBI
    params.outputflag = 0;
    result = gurobi(model, params);
    x = reshape(result.x(1:n*m), [n, m]);

else
    %% define variables
    x = sdpvar(n,m);

    %% Define constraints
    Constraints = [x >= zeros(n,m)];
    for i = 1:n
        Constraints = [Constraints, sum(x(i,:)) <= theta(i)];
    end

    %% Define an objective
    g = ones(N,1)*sum(x)-S;
    f = trace(c'*x0)+trace(a'*x0.^2)+trace((c+2*a.*x0)'*(x-x0));
    phi = 0.5*(x-x0)'*H*(x-x0);
    Objective = f + trace(lambda'*(y-g)) + rho/2*norm(y-g,'fro')^2 + phi;

    %% solve by GUROBI solver
    ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 0, 'gurobi.timelimit', maxitime);
    sol = optimize(Constraints, Objective, ops);
    x = value(x);

end
end