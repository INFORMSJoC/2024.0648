function [fval, y, z] = ALDM_update_y(S, c, x, lambda, rho, d, alpha)

% given lambda, fix x.
% min L(\barx,y,\barlambda)
% s.t. (y,z) \in Omega.
% The above problem is equivalent to the following 0-1 linear knapsack
% problem:
% max sum^N(r_i-q_i)*z_i
% s.t. sum^N(p_i*z_i) <= alpha, z \in {0,1}^N.

[~, m] = size(c); N = size(S,1); [r, q, y] = deal(zeros(N,m));
K = floor(N*alpha); z = zeros(N,1);
g = ones(N,1)*sum(x)-S; w = g - lambda/rho;

%% compute r and q
for i = 1:N
    for j = 1:m
        if w(i,j) < 0
            r(i,j) = w(i,j)^2; y(i,j) = 0;
        else
            r(i,j) = 0; y(i,j) = w(i,j);
        end
        if w(i,j) < d(i,j)
            q(i,j) = (d(i,j)-w(i,j))^2;
        else
            q(i,j) = 0;
        end
    end
end
R = sum(r,2); Q = sum(q,2);

% rank the sequence {r_i-q_i}
f = R-Q;
[B,inx] = sort(f,'descend');
fval = sum(B(1:K));
z(inx(1:K),1) = 1;
for ii = inx(1:K)
    for jj = 1:m
        if q(ii,jj) == 0
            y(ii,jj) = w(ii,jj);
        else
            y(ii,jj) = d(ii,jj);
        end
    end
end
end