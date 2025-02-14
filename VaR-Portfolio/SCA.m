function [x, time, iter] = SCA(S, alpha, beta, sigma, mu, u, R, opts)

fprintf('***************** SCA for DC approximation ***************** \n');

%% parameter setting
[n, ~] = size(u); N = size(S,1); maxiter = 1e2;

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
if isfield(opts,'x0')
    x0 = opts.x0;
else
    x0 = randn(n,1);
end
if isfield(opts, 'tol')
    tol = opts.tol;
else
    tol = 1e-6;
end

xk = x0;
fval = beta*x0'*sigma*x0-mu*x0; time = 0;
CN = zeros(N,1); inx_set = zeros(N,1);
t = 1e-3;
for iter = 1 : maxiter

    fval_old = fval;

    %% compute subgradient of \sum_{i=1}^N \max{C(x,\hat{\xi}^i),0} at xk
    C = R*ones(N,1)-S*xk;
    for i = 1:N
        [CN(i), inx_set(i)] = max(C(i,:));
    end

    gk = zeros(n,1); gval = 0;
    for i = 1:N
        if CN(i) >= 0
            sk = -S(i,:)';
            gk = gk + sk; gval = gval + CN(i);
        end
    end
    if strcmp(solver,'gurobi') == 1
        model.A = sparse([ones(1,n) sparse(1,N);...
            -gk' ones(1,N);...
            -S -speye(N)]);
        model.Q =[beta*sigma sparse(n,N); sparse(N,n) sparse(N,N)];
        model.obj = [-mu zeros(1,N)];
        model.lb = [zeros(1,n) zeros(1,N)];
        model.ub = [u' inf(1,N)];
        model.modelsense = 'Min';
        model.rhs = [1 N*alpha*t+gval-gk'*xk -R*ones(1,N)-t*ones(1,N)];
        sign = ['='];
        for i=2:(N+2)
            sign=[sign,'<'];
        end
        model.sense = sign;

        %% Solve the problem using GUROBI
        params.outputflag = 0;
        params.timelimit = maxitime;
        result = gurobi(model, params);
        xk = result.x(1:n);
        time = time + result.runtime;
    else
        %% define variables
        x = sdpvar(n,1); z = sdpvar(N,1); t = 1e-3;

        %% intial point
        assign(x, x0);

        %% Define constraints
        Constraints = [sum(z) - (gval + trace(gk'*(x-xk))) <= N*alpha*t, ...
            z >= zeros(N,1)];
        for i = 1:n
            Constraints = [Constraints, ones(n,1)'*x == 1, zeros(n,1) <= x <= u];
        end
        Constraints = [Constraints, R*ones(N,1)-S*x <= z - t*ones(N,1)];


        %% Define an objective
        Objective = beta*x'*sigma*x-mu*x;

        %% solve by GUROBI solver
        ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 1, 'gurobi.timelimit', maxitime);
        sol = optimize(Constraints, Objective, ops);
        time = time + sol.solvertime;

        %% report iterate information
        xk = value(x);
    end

    fval = beta*xk'*sigma*xk-mu*xk;

    if abs(fval - fval_old) <= tol || time > 1800
        break;
    end

end
x = xk;
end