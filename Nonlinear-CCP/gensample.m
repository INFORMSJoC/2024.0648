function xi = gensample(N, n, m)

xi_mean = zeros(1, n);
xi_covariance = 0.5 * ones(m, m);

for j = 1:n
    xi_mean(j) = j / n;
end

for i = 1:m
    xi_covariance(i, i) = 1;
end

xi_cov_chol = chol(xi_covariance, 'lower');

xi_mean_mat = zeros(m, n);

for j = 1:n
    xi_mean_mat(:, j) = xi_mean(j) * ones(m, 1);
end

xi = zeros(m, n, N);
xi_tmp = randn(m, n, N);

for samp = 1:N
    xi(:,:,samp) = xi_cov_chol * xi_tmp(:,:,samp) + xi_mean_mat;
end
xi = permute(xi, [3, 2, 1]);
end