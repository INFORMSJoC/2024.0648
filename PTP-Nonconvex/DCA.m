function [x, time, iter] = DCA(S, c, a, theta, alpha, opts)

fprintf('***************** DCA ***************** \n');

%% parameter setting
[n, m] = size(c); N = size(S,1);
M = ceil((1-alpha)*N); T = N - M; maxiter = 1e2;
s = reshape(S,[1,N*m]); c1 = reshape(c,[1,n*m]);
a1 = reshape(a,[1,n*m]);

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
    x0 = randn(n,m);
end
if isfield(opts, 'tol')
    tol = opts.tol;
else
    tol = 1e-6;
end

xk = x0; fval = trace(c'*x0)+trace(a'*x0.^2); time = 0;

for iter = 1 : maxiter

    fval_old = fval;
    %% compute subgradient of C(x,\xi^l) for l = 1,...,N at xk
    C =  S - ones(N,1)*sum(xk);
    [CN, inx_set]= max(C,[],2);
    %% compute subgradient of H(x) at xk
    [mz, I] = maxk(CN, N-M); mz_fval = sum(mz);
    gk = zeros(n,m);
    for i = 1:size(I,1)
        inx = inx_set(I(i)); sk = zeros(n,m); sk(:,inx) = -1;
        gk = gk + sk;
    end

    if strcmp(solver,'gurobi') == 1
        axk = a1.*xk(:)';
        gk = gk(:);
        %% Model parameters in GUROBI
        model.A = sparse([-gk' sparse(1,N) ones(1,N) T+1;...
            sparse(N,n*m) speye(N) -speye(N) -ones(N,1);...
            -kron(speye(m),ones(N,n)) -kron(ones(m,1),speye(N)) sparse(N*m,N) zeros(N*m,1);...
            kron(ones(1,m),speye(n)) sparse(n,N) sparse(n,N) zeros(n,1)]);
        model.obj = [c1+2*axk zeros(1,2*N+1)];
        model.lb = [zeros(1,n*m) -inf(1,N) zeros(1,N) -inf];
        model.ub = inf(1,n*m+2*N+1);
        model.modelsense = 'Min';
        model.rhs = [mz_fval-gk'*xk(:) zeros(1,N) -s theta'];
        model.sense = '<';

        %% Solve the problem using GUROBI
        params.outputflag = 0;
        result = gurobi(model, params);
        time = time + result.runtime;
        xk = reshape(result.x(1:n*m), [n, m]);
    else
        %% define variables
        x = sdpvar(n,m); z = sdpvar(N,1); lambda = sdpvar(N,1); muv = sdpvar(1);

        %% intial point
        assign(x, x0);

        %% Define constraints
        Constraints = [
            sum(lambda) + (T+1)*muv - mz_fval - trace(gk'*(x-xk)) <= 0,...
            z - lambda - muv*ones(N,1) <= zeros(N,1), lambda >= zeros(N,1), x >= zeros(n,m)];
        for i = 1:n
            Constraints = [Constraints, sum(x(i,:)) <= theta(i)];
        end
        for j = 1:m
            Constraints = [Constraints, S(:,j)-sum(x(:,j))*ones(N,1) <= z];
        end

        %% Define an objective
        b = 2*a.*xk;
        Objective = trace(c'*x)+trace(b'*x);

        %% solve by GUROBI solver
        ops = sdpsettings('solver', solver, 'verbose', 0, 'usex0', 1, 'gurobi.timelimit', maxitime);
        sol = optimize(Constraints, Objective, ops);
        time = time + sol.solvertime;

        %% report iterate information
        xk = value(x);
    end

    fval = trace(c'*xk) + trace(a'*xk.^2);

    fprintf('Iternum: %d, fval: %.4f\n', iter, fval);

    if abs(fval - fval_old)/max(1,fval_old) <= tol || time > 1800
        break;
    end
end
x = xk;
end