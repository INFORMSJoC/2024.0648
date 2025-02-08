function [x, time, iter] = SCA(S, c, a, theta, alpha, eps, opts)

fprintf('***************** SCA for DC approximation ***************** \n');

%% parameter setting
[n, m] = size(c); N = size(S,1); maxiter = 1e2;
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

t = eps;
for iter = 1 : maxiter

    fval_old = fval;

    %% compute subgradient of \sum_{i=1}^N \max{C(x,\hat{\xi}^i),0} at xk
    C =  S - ones(N,1)*sum(xk);
    [CN, inx_set]= max(C,[],2);

    gk = zeros(n,m); gval = 0;
    for i = 1:N
        if CN(i) >= 0
            inx = inx_set(i); sk = zeros(n,m); sk(:,inx) = -1;
            gk = gk + sk; gval = gval + CN(i);
        end
    end
    if strcmp(solver,'gurobi') == 1
        %% Model parameters in GUROBI
        axk = a1.*xk(:)';
        gk = gk(:);
        model.A = sparse([-gk' ones(1,N); kron(ones(1,m),speye(n)) sparse(n,N);...
            -kron(speye(m),ones(N,n)) -kron(ones(m,1),speye(N))]);
        model.lb = [zeros(1,n*m) zeros(1,N)];
        model.ub = inf(1,n*m+N);
        model.obj = [c1+2*axk zeros(1,N)];
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
            time = time + result.runtime;
        end
    end
    fval = trace(c'*xk) + trace(a'*xk.^2);

    fprintf('Iternum: %d, fval: %.4f\n', iter, fval);
    if abs(fval - fval_old) <= tol || time > 1800
        break;
    end
end
x = xk;
end