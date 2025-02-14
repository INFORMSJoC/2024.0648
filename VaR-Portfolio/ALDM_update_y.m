function [fval, y, z] = ALDM_update_y(S, x, R, lambda, rho, d, alpha)
% xi : R^n
% x : R^n
% S : R^n*N
% w : R^1*N
N = size(S,1);
[~, m] = size(x);
K = floor(N*alpha);
z = zeros(N,1);

w = [];
for i = 1:N
    w = [w; S(i,:)*x-R-lambda(i)/rho];
end

r = zeros(N,m);
q = zeros(N,m);
y = zeros(N,m);
for i = 1:N
    for j = 1:m
        if w(i,j) < 0
            r(i,j) = w(i,j)^2;
            y(i,j) = 0;
        else
            r(i,j) = 0;
            y(i,j) = w(i,j);
        end

        if w(i,j) < d(i,j)
            q(i,j) = (d(i,j)-w(i,j))^2;
        else
            q(i,j) = 0;
        end
    end
end
% R = sum(r');
% Q = sum(q');

f = r-q;
[B,IX] = sort(f,'descend');
fval = sum(B(1:K));
z(IX(1:K),1) = 1;
for ii = IX(1:K)
    for jj = 1:m
        if q(ii,jj) == 0
            y(ii,jj) = w(ii,jj);
        else
            y(ii,jj) = d(ii,jj);
        end
    end
end
end