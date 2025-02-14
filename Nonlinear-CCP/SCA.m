function [x, time, iter] = SCA(S, U, alpha, t, opts)

fprintf('***************** SCA for DC approximation ***************** \n');

%% parameter setting
[N, n, m] = size(S);
maxiter = 2*1e2;

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
if isfield(opts,'x0')
    x0 = opts.x0;
else
    x0 = randn(n,1);
end
if isfield(opts, 'tol')
    tol = opts.tol;
else
    tol = 1e-4;
end

xk = x0;
fval = -sum(x0);
time = 0;

for iter = 1 : maxiter
    fval_old = fval;
    %% compute subgradient of \sum_{i=1}^N \max{C(x,\hat{\xi}^i),0} at xk
    for j = 1:m
        C(:,j) = (S(:,:,j).^2)*(xk.^2) - U;
    end
    [CN, inx_set]= max(C,[],2);
    %% compute subgradient of H(x) at xk
    gk = zeros(n,1);
    gval = 0;
    for i = 1:N
        if CN(i) >= 0
            inx = inx_set(i);
            sk = 2*(S(i,:,inx).^2)'.*xk;
            gk = gk + sk; gval = gval + CN(i);
        end
    end
    
%% CVX
    cvxtime = tic;
    cvx_begin quiet
        variables x(n,1) z(N,1)
        Constraints = [sum(z) - (gval + gk'*(x-xk)) <= N*alpha*t, x >= zeros(n,1),...
        z >= zeros(N,1)];
        for j = 1:m
            Constraints = [Constraints, (S(:,:,j).^2)*(x.^2) - U*ones(N,1) + t*ones(N,1)  <= z];
        end
    
        Objective = -sum(x);
        cvx_solver gurobi
        cvx_solver_settings('OptimalityTol', tol)
        cvx_solver_settings('TimeLimit', maxitime)
        minimize(Objective)
        subject to
            Constraints
    cvx_end

    iter_time = toc(cvxtime);
    xk = x; 
    time = time + iter_time;

    fval = -sum(xk);
    fprintf('Iternum: %d, fval: %.4f\n', iter, fval);

    if abs(fval - fval_old)/max(1, abs(fval_old)) <= tol || time > 1800
        break;
    end
end
x = xk;
end