function x = ALDM_update_x_g(beta, S, y, u, x0, sigma, mu, R, rho, lambda, H, opts)

[n, ~] = size(u); N = size(S,1);
%% parameter setting

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
    model.A = sparse(ones(1,n));
    model.lb = zeros(1,n);
    model.ub = u';
    model.Q =sparse(beta*sigma+rho/2*(S'*S)+0.5*H);
    model.obj = -mu-lambda'*S-rho*(y+R*ones(N,1))'*S-x0'*H;
    model.modelsense = 'Min';
    model.rhs = 1;
    model.sense = '=';

    %% Solve the problem using GUROBI
    params.outputflag = 0;
    params.timelimit = maxitime;
    result = gurobi(model, params);
    x = result.x;
else
    %% define variables
    x = sdpvar(n,1);

    %% Define constraints
    Constraints = [ones(n,1)'*x == 1, zeros(n,1) <= x <= u];

    %% Define an objective
    f = beta*x'*sigma*x-mu*x;
    g = S*x-R*ones(N,1);
    L = sum(lambda.*(y-g))+rho/2*norm(y-g,'fro')^2;
    phi = 0.5*(x-x0)'*H*(x-x0);
    Objective = f+L+phi;

    %% solve by GUROBI solver
    ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 0, 'gurobi.timelimit', maxitime);
    sol = optimize(Constraints, Objective, ops);
    x = value(x);
end