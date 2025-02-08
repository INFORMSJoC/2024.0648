function [x, time, iter] = DCA(S, U, alpha, opts)

fprintf('***************** DCA ***************** \n');

%% parameter setting
[N, n, m] = size(S);
M = ceil((1-alpha)*N); T = N - M; maxiter = 2*1e2;

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
    x0 = randn(n,m);
end
if isfield(opts, 'tol')
    tol = opts.tol;
else
    tol = 1e-4;
end

xk = x0; fval = -sum(x0); time = 0;

for iter = 1 : maxiter
    
    fval_old = fval;
    %% compute subgradient of C(x,\xi^l) for l = 1,...,N at xk
    for j = 1:m
        C(:,j) = (S(:,:,j).^2)*(xk.^2) - U;
    end
    [CN, inx_set]= max(C,[],2);
    %% compute subgradient of H(x) at xk
    [mz, I] = maxk(CN, N-M);
    mz_fval = sum(mz);
    gk = zeros(n,1);
    for i = 1:size(I,1)
        inx = inx_set(I(i));
        sk = 2*(S(I(i),:,inx).^2)'.*xk;
        gk = gk + sk;
    end

%% CVX
    cvxtime = tic;
    cvx_begin quiet
        variables x(n) z(N) lambda(N) muv(1)
        Constraints = [
            sum(lambda) + (T+1)*muv - mz_fval - gk'*(x-xk) <= 0,...
            z - lambda - muv*ones(N,1) <= zeros(N,1), lambda >= zeros(N,1), x >= zeros(n,1)];
        for j = 1:m
            Constraints = [Constraints, (S(:,:,j).^2)*(x.^2) - U <= z];
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

    if abs(fval - fval_old)/max(1,fval_old) <= tol || time > 1800
        break;
    end
end
x = xk;
end