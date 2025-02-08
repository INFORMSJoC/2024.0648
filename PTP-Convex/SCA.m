function [x, time, iter,ii] = SCA(S, c, theta, alpha, opts)

fprintf('***************** SCA for DC approximation ***************** \n');

%% parameter setting
[n, m] = size(c); N = size(S,1); maxiter = 1e2;
s = reshape(S,[1,N*m]); c1 = reshape(c,[1,n*m]);
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

xk = x0;
fval = trace(c'*x0);
time = 0;
C = zeros(N,m); CN = zeros(N,1); inx_set = zeros(N,1);
t = 1e-10;
for iter = 1 : maxiter

    fval_old = fval;

    %% compute subgradient of \sum_{i=1}^N \max{C(x,\hat{\xi}^i),0} at xk
    for j = 1:m
        C(:,j) = S(:,j)-sum(xk(:,j))*ones(N,1);
    end
    for i = 1:N
        [CN(i), inx_set(i)] = max(C(i,:));
    end

    gk = zeros(n,m); gval = 0;
    for i = 1:N
        if CN(i) >= 0
            inx = inx_set(i); sk = zeros(n,m); sk(:,inx) = -1;
            gk = gk + sk; gval = gval + CN(i);
        end
    end
    if strcmp(solver,'gurobi') == 1
        %% Model parameters in GUROBI
        gk = gk(:);
        model.A = sparse([-gk' ones(1,N); kron(ones(1,m),speye(n)) sparse(n,N);...
            -kron(speye(m),ones(N,n)) -kron(ones(m,1),speye(N))]);
        model.lb = [zeros(1,n*m) zeros(1,N)];
        model.ub = inf(1,n*m+N);
        model.obj = [c1 zeros(1,N)];
        model.modelsense = 'Min';
        model.rhs = [N*alpha*t+gval-gk'*xk(:) theta' -s-ones(1,N*m)*t];
        model.sense = '<';

        %% Solve the problem using GUROBI
        params.outputflag = 0;
        params.timelimit = maxitime;
        result = gurobi(model, params);
        if ~strcmp(result.status(1:7),'OPTIMAL')&&~strcmp(result.status(1:7),'SUBOPTI')
            xk = zeros(n,m);
        else
            xk = reshape(result.x(1:n*m), [n, m]);
            time = result.runtime;
        end
    end
    fval = trace(c'*xk);

    fprintf('Iternum: %d, fval: %.4f\n', iter, fval);
    if fval == 0
        ii=1;
        break;
    else
        ii=0;
    end

    if abs(fval - fval_old) <= tol || time > 1800
        break;
    end
end
x = xk;
end