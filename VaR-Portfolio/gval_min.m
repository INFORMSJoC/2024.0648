function d = gval_min(S, u, R)

[n, ~] = size(u); N = size(S,1);  d = zeros(N,1);

model.A = sparse(ones(1,n));
model.lb = zeros(1,n);
model.ub = u';
model.rhs = 1;
model.sense = '=';
for i = 1:N
    model.obj = S(i,:);
    model.modelsense = 'Min';
    %% Solve the problem using GUROBI
    params.outputflag = 0;
    result = gurobi(model, params);
    x = result.x;
    d(i) = S(i,:)*x-R;
end
end