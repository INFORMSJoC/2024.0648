function [x, time,iter] = pDCA(S, alpha, beta, sigma, mu, u, R, rho, opts)

fprintf('***************** pDCA ***************** \n');

%% parameter setting
[n, ~] = size(u); N = size(S,1);
M = ceil((1-alpha)*N); T = N - M; maxiter = 1e2;

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

xk = x0; fval = beta*x0'*sigma*x0-mu*x0; time = 0;
CN = zeros(N,1); inx_set = zeros(N,1);

for iter = 1 : maxiter
    fval_old = fval;
    %% compute subgradient of C(x,\xi^l) for l = 1,...,N at xk
    C = R*ones(N,1)-S*xk;

    for i = 1:N
        [CN(i), inx_set(i)] = max(C(i,:));
    end
    %% compute subgradient of H(x) at xk
    [mz, I] = maxk(CN, N-M); mz_fval = sum(mz);
    gk = zeros(n,1);
    for i = 1:size(I,1)
        inx = I(i); sk = -S(inx,:)';
        gk = gk + sk;
    end
    if strcmp(solver,'gurobi') == 1
        model.A = sparse([ones(1,n) sparse(1,2*N+1);...
            -gk' sparse(1,N) ones(1,N) T+1;...
            sparse(N,n) speye(N) -speye(N) -ones(N,1);...
            -S -speye(N) sparse(N,N) sparse(N,1)]);
        model.Q =[beta*sigma+(rho/2)*speye(n) sparse(n,2*N+1); sparse(2*N+1,n) sparse(2*N+1,2*N+1)];
        model.obj = [-mu-rho*xk' zeros(1,2*N+1)];
        model.lb = [zeros(1,n) -inf(1,N) zeros(1,N) -inf];
        model.ub = [u' inf(1,2*N+1)];
        model.modelsense = 'Min';
        model.rhs = [1 mz_fval-gk'*xk zeros(1,N) -R*ones(1,N)];
        sign = ['='];
        for i=2:(2*N+2)
            sign=[sign,'<'];
        end
        model.sense = sign;

        %% Solve the problem using GUROBI
        params.outputflag = 0;
        result = gurobi(model, params);
        time = time + result.runtime;
        xk = result.x(1:n);
    else
        %% define variables
        x = sdpvar(n,m); z = sdpvar(N,1);
        lambda = sdpvar(N,1); muv = sdpvar(1);

        %% intial point
        assign(x, x0); assign(z, z0); assign(lambda, lambda0); assign(muv, mu0);

        %% Define constraints
        Constraints = [
            -ones(N,1)'*lambda - (T+1)*muv - mz_fval- trace(gk'*(x-xk)) <= 0,...
            z + lambda + muv*ones(N,1) <= zeros(N,1), lambda <= zeros(N,1),...
            ones(n,1)'*x == 1, zeros(n,1) <= x <= u];
        Constraints = [Constraints, R*ones(N,1)-S*x <= z];
        %% Define an objective
        Objective = beta*x'*sigma*x-mu*x+rho/2*norm(x-xk,'fro')^2;

        %% solve by GUROBI solver
        ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 1, 'gurobi.timelimit', maxitime);
        sol = optimize(Constraints, Objective, ops);
        time = time + sol.solvertime;
        %% report iterate information
        xk = value(x);
    end

    fval = beta*xk'*sigma*xk-mu*xk;
    rho = rho/4;
    if abs(fval - fval_old)/max(1,fval_old) <= tol || time > maxitime
        break;
    end
end
x = xk;
end