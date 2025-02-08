%%%%%%%%%%%%%%%%%%% Tests on Portfolio Selection Problem %%%%%%%%%%%%%%%%%%%%%
% Portfolio Selection Problem:
% min beta*x^T*Sigma*x-mu^T*x
% s.t. P(xi^T*x>=R)>=1-alpha,
%      e^T*x=1, 0<=x<=u.
clear; clc;
addpath(genpath(pwd));

fprintf('***************** Tests on Portfolio Selection Problem *****************\n');
a = 10; % risk level(%)
%% choose the method
IP = 1; CVa = 1; Bi = 1; DC = 1; pDC = 1; pDC1 = 1; pDC2 = 1; AL = 1; SC = 1; repeat_num = 5;

for n = [100, 200, 300, 400]
    N = 3*n;
    
    %% collect iteration information
    [iter_ALDM_collect, iter_DC_collect, iter_pDC_collect, iter_pDC1_collect, iter_pDC2_collect, iter_SCA_collect, iter_SCA1_collect] = deal(zeros(repeat_num,1));
    [fval_MIP_collect, fval_CVaR_collect, fval_Bi_collect, fval_DC_collect, fval_pDC_collect, fval_pDC1_collect, fval_pDC2_collect, fval_SCA_collect, fval_SCA1_collect, fval_ALDM_collect] = deal(zeros(repeat_num,1));
    [prob_MIP_collect, prob_CVaR_collect, prob_Bi_collect, prob_DC_collect, prob_pDC_collect, prob_pDC1_collect, prob_pDC2_collect, prob_SCA_collect, prob_SCA1_collect, prob_ALDM_collect] = deal(zeros(repeat_num,1));
    [time_MIP_collect, time_CVaR_collect, time_Bi_collect, time_DC_collect, time_pDC_collect, time_pDC1_collect, time_pDC2_collect, time_SCA_collect, time_SCA1_collect, time_ALDM_collect] = deal(zeros(repeat_num, 1));
    [ttime_MIP_collect, ttime_CVaR_collect, ttime_Bi_collect, ttime_DC_collect, ttime_pDC_collect, ttime_pDC1_collect, ttime_pDC2_collect, ttime_SCA_collect, ttime_SCA1_collect, ttime_ALDM_collect] = deal(zeros(repeat_num, 1));

    for iter = 1:repeat_num
        %% load data
        fid = fopen(['data_',num2str(a),'_',num2str(n),'_',num2str(N),'_',num2str(iter),'.txt'],'r');
        fout = fopen('new.txt','w');
        i = 0;
        while ~feof(fid)
            tline = fgetl(fid);
            if i == 0
                i = i + 1;
                continue
            else
                fprintf(fout,'%s\n', tline);
            end
            i = i + 1;
        end
        fclose(fid);
        fclose(fout);
        data = importdata('new.txt');
        %% parameter setting
        beta = 2;       % risk aversion factor
        u = 0.5*ones(n,1);  % upper bound vector for x
        R = 0.0002;       %  return level 0.0002
        alpha = a/100;        % risk level

        %% sample based method
        train_samples = data((n+1):(n+N),1:n);
        sigma = data(1:n,1:n); mu = mean(train_samples);

        %% parameter setting
        solver = 'gurobi'; maxitime = 1800; tol = 1e-6;

        fprintf('***************** num of repeat: %d ***************** \n', iter);
        %% MIP
        if IP == 1
            opts = struct('solver', solver, 'maxitime', maxitime); upper = 1e6;
            tic; [x_MIP, time_MIP] = MIP(train_samples, upper, alpha, u, beta, sigma, mu, R, opts); ttime_MIP = toc;
            fval_MIP = beta*x_MIP'*sigma*x_MIP-mu*x_MIP; prob_MIP = risk_level(train_samples, x_MIP, R);
            fprintf('MIP: %f \n', fval_MIP);
            fval_MIP_collect(iter) = fval_MIP; prob_MIP_collect(iter) = prob_MIP;
            ttime_MIP_collect(iter) = ttime_MIP; time_MIP_collect(iter) = time_MIP;
        end

        %% CVaR
        if CVa == 1
            opts = struct('solver', solver, 'maxitime', maxitime);
            tic; [x0, time_CVaR] = CVaR(train_samples, u, alpha, beta, sigma, mu, R, opts); ttime_CVaR = toc;
            fval_CVaR = beta*x0'*sigma*x0-mu*x0; prob_CVaR = risk_level(train_samples, x0, R);  fprintf('CVaR: %f \n', fval_CVaR);
            fval_CVaR_collect(iter) = fval_CVaR; prob_CVaR_collect(iter) = prob_CVaR;
            ttime_CVaR_collect(iter) = ttime_CVaR; time_CVaR_collect(iter) = time_CVaR;
        end

        %% Bisection Based CVaR
        if Bi == 1
            opts = struct('solver', solver, 'maxitime', maxitime);
            tic; [x_Bi, time_Bi] = BiCVaR(train_samples, u, alpha, beta, sigma, mu, R, opts); ttime_Bi = toc;
            fval_Bi = beta*x_Bi'*sigma*x_Bi-mu*x_Bi; prob_Bi = risk_level(train_samples, x_Bi, R); fprintf('BiCVaR: %f \n', fval_Bi);
            fval_Bi_collect(iter) = fval_Bi; prob_Bi_collect(iter) = prob_Bi;
            ttime_Bi_collect(iter) = ttime_Bi; time_Bi_collect(iter) = time_Bi;
        end

        %% Augmented Lagrangian Decomposition method (ALDM)
        if AL == 1
            solver = 'gurobi'; init = 1; % init = 1: initialization by CVaR
            if init == 1
                %% compute lower bound of each g(x,\xi^i) for i = 1,\dots,N by d_i = min{g(x,\xi^i): x \in X}
                d = gval_min(train_samples, u, R);
                opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol, 'd', d);
            else
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol, 'd', d);
            end
            tic; [x_ALDM,iter_ALDM,time_ALDM] = ALDM(beta, train_samples, alpha, R, u, sigma, mu, opts); ttime_ALDM = toc;
            fval_ALDM = beta*x_ALDM'*sigma*x_ALDM-mu*x_ALDM; prob_ALDM = risk_level(train_samples, x_ALDM, R);
            fprintf('ALDM: %f \n', fval_ALDM);
            iter_ALDM_collect(iter) = iter_ALDM;
            fval_ALDM_collect(iter) = fval_ALDM; prob_ALDM_collect(iter) = prob_ALDM;
            ttime_ALDM_collect(iter) = ttime_ALDM; time_ALDM_collect(iter) = time_ALDM;
        end

        %% SCA
        if SC == 1
            solver = 'gurobi'; init = 1; % init = 1: initialization by CVaR
            if init == 1
                opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
            else
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
            end
            tic; [x_SCA, time_SCA, iter_SCA] = SCA(train_samples, alpha, beta, sigma, mu, u, R, opts); ttime_SCA = toc;
            fval_SCA = beta*x_SCA'*sigma*x_SCA-mu*x_SCA; prob_SCA = risk_level(train_samples, x_SCA, R);
            fprintf('SCA: %f\n', fval_SCA);
            iter_SCA_collect(iter) = iter_SCA;
            fval_SCA_collect(iter) = fval_SCA; prob_SCA_collect(iter) = prob_SCA;
            ttime_SCA_collect(iter) = ttime_SCA; time_SCA_collect(iter) = time_SCA;
        end

        %% DC algorithm
        if DC == 1
            solver = 'gurobi'; init = 1; %% init = 1: initialization by CVaR
            if init == 1
                opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
            else
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
            end
            tic; [x_DC, time_DC, iter_DC] = DCA(train_samples, alpha, beta, sigma, mu, u, R, opts);  ttime_DC = toc;
            fval_DC = beta*x_DC'*sigma*x_DC-mu*x_DC; prob_DC = risk_level(train_samples, x_DC, R);
            iter_DC_collect(iter) = iter_DC; fprintf('DC: %f \n', fval_DC);
            fval_DC_collect(iter) = fval_DC; prob_DC_collect(iter) = prob_DC;
            ttime_DC_collect(iter) = ttime_DC; time_DC_collect(iter) = time_DC;
        end

        %% proximal DC algorithm

        if pDC == 1
            solver = 'gurobi';
            rho = 0.1; % parameter of the proximal term
            init = 1; % init = 1: initialization by CVaR
            if init == 1
                opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
            else
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
            end
            tic; [x_pDC, time_pDC, iter_pDC] = pDCA(train_samples, alpha, beta, sigma, mu, u, R, rho, opts);  ttime_pDC = toc;
            fval_pDC = beta*x_pDC'*sigma*x_pDC-mu*x_pDC; prob_pDC = risk_level(train_samples, x_pDC, R);
            iter_pDC_collect(iter) = iter_pDC; fprintf('pDC (rho = 0.1): %f,time: %f \n', fval_pDC, ttime_pDC);
            fval_pDC_collect(iter) = fval_pDC; prob_pDC_collect(iter) = prob_pDC;
            ttime_pDC_collect(iter) = ttime_pDC; time_pDC_collect(iter) = time_pDC;
        end

        if pDC1 == 1
            solver = 'gurobi';
            rho = 1; % parameter of the proximal term
            init = 1; % init = 1: initialization by CVaR
            if init == 1
                opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
            else
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
            end
            tic; [x_pDC1, time_pDC1, iter_pDC1] = pDCA(train_samples, alpha, beta, sigma, mu, u, R, rho, opts);  ttime_pDC1 = toc;
            fval_pDC1 = beta*x_pDC1'*sigma*x_pDC1-mu*x_pDC1; prob_pDC1 = risk_level(train_samples, x_pDC1, R);
            iter_pDC1_collect(iter) = iter_pDC1; fprintf('pDC (rho = 1): %f,time: %f \n', fval_pDC1, ttime_pDC1);
            fval_pDC1_collect(iter) = fval_pDC1; prob_pDC1_collect(iter) = prob_pDC1;
            ttime_pDC1_collect(iter) = ttime_pDC1; time_pDC1_collect(iter) = time_pDC1;
        end

        if pDC2 == 1
            solver = 'gurobi';
            rho = 10; % parameter of the proximal term
            init = 1; % init = 1: initialization by CVaR
            if init == 1
                opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
            else
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
            end
            tic; [x_pDC2, time_pDC2, iter_pDC2] = pDCA(train_samples, alpha, beta, sigma, mu, u, R, rho, opts);  ttime_pDC2 = toc;
            fval_pDC2 = beta*x_pDC2'*sigma*x_pDC2-mu*x_pDC2; prob_pDC2 = risk_level(train_samples, x_pDC2, R);
            iter_pDC2_collect(iter) = iter_pDC2; fprintf('pDC (rho = 10): %f,time: %f \n', fval_pDC2, ttime_pDC2);
            fval_pDC2_collect(iter) = fval_pDC2; prob_pDC2_collect(iter) = prob_pDC2;
            ttime_pDC2_collect(iter) = ttime_pDC2; time_pDC2_collect(iter) = time_pDC2;
        end
    end

    save(strcat('savePortData_',datestr(now,'mm-dd_HH-MM'),'a=',num2str(a),'_n=',num2str(n),'_N=',num2str(N)));
    %% print information
    fprintf('num of assets = %d, num of samples =%d\n', n, N);
    fid = fopen(['Port_date',datestr(now,'mm-dd_HH-MM'),'_a=',num2str(alpha),'n=',num2str(n),'_N=',num2str(N),'.txt'],'wt');
    if IP == 1
        [fval_MIP_ave, fval_MIP_min, fval_MIP_max] = post_processing(fval_MIP_collect);
        [prob_MIP_ave, prob_MIP_min, prob_MIP_max] = post_processing(prob_MIP_collect);
        [time_MIP_ave, time_MIP_min, time_MIP_max] = post_processing(time_MIP_collect);
        fprintf(fid,'MIP fval: %.6f\t time: %.4f\t prob: %.4f\n', fval_MIP_ave, time_MIP_ave, prob_MIP_ave);
    end
    if CVa == 1
        [fval_CVa_ave, fval_CVa_min, fval_CVa_max] = post_processing(fval_CVaR_collect);
        [prob_CVa_ave, prob_CVa_min, prob_CVa_max] = post_processing(prob_CVaR_collect);
        [time_CVa_ave, time_CVa_min, time_CVa_max] = post_processing(time_CVaR_collect);
        fprintf(fid,'CVaR fval: %.6f\t time: %.4f\t prob: %.4f\n', fval_CVa_ave, time_CVa_ave, prob_CVa_ave);
    end
    if Bi == 1
        [fval_Bi_ave, fval_Bi_min, fval_Bi_max] = post_processing(fval_Bi_collect);
        [prob_Bi_ave, prob_Bi_min, prob_Bi_max] = post_processing(prob_Bi_collect);
        [time_Bi_ave, time_Bi_min, time_Bi_max] = post_processing(time_Bi_collect);
        fprintf(fid,'BiCVaR fval: %.6f\t time: %.4f\t prob: %.4f\n', fval_Bi_ave, time_Bi_ave, prob_Bi_ave);
    end
    if DC == 1
        [fval_DC_ave, fval_DC_min, fval_DC_max] = post_processing(fval_DC_collect);
        [prob_DC_ave, prob_DC_min, prob_DC_max] = post_processing(prob_DC_collect);
        [time_DC_ave, time_DC_min, time_DC_max] = post_processing(time_DC_collect);
        [iter_DC_ave, iter_DC_min, iter_DC_max] = post_processing(iter_DC_collect);
        fprintf(fid,'DCA fval: %.6f\t time: %.4f\t prob: %.4f\t iter: %.2f\n', fval_DC_ave, time_DC_ave, prob_DC_ave, iter_DC_ave);
    end
    if pDC == 1
        [fval_pDC_ave, fval_pDC_min, fval_pDC_max] = post_processing(fval_pDC_collect);
        [prob_pDC_ave, prob_pDC_min, prob_pDC_max] = post_processing(prob_pDC_collect);
        [time_pDC_ave, time_pDC_min, time_pDC_max] = post_processing(time_pDC_collect);
        [iter_pDC_ave, iter_pDC_min, iter_pDC_max] = post_processing(iter_pDC_collect);
        fprintf(fid,'pDCA fval (rho = 0.1): %.6f\t time: %.4f\t prob: %.4f\t iter: %.2f\n', fval_pDC_ave, time_pDC_ave, prob_pDC_ave, iter_pDC_ave);
    end
    if pDC1 == 1
        [fval_pDC1_ave, fval_pDC1_min, fval_pDC1_max] = post_processing(fval_pDC1_collect);
        [prob_pDC1_ave, prob_pDC1_min, prob_pDC1_max] = post_processing(prob_pDC1_collect);
        [time_pDC1_ave, time_pDC1_min, time_pDC1_max] = post_processing(time_pDC1_collect);
        [iter_pDC1_ave, iter_pDC1_min, iter_pDC1_max] = post_processing(iter_pDC1_collect);
        fprintf(fid,'pDCA fval (rho = 1): %.6f\t time: %.4f\t prob: %.4f\t iter: %.2f\n', fval_pDC1_ave, time_pDC1_ave, prob_pDC1_ave, iter_pDC1_ave);
    end
    if pDC2 == 1
        [fval_pDC2_ave, fval_pDC2_min, fval_pDC2_max] = post_processing(fval_pDC2_collect);
        [prob_pDC2_ave, prob_pDC2_min, prob_pDC2_max] = post_processing(prob_pDC2_collect);
        [time_pDC2_ave, time_pDC2_min, time_pDC2_max] = post_processing(time_pDC2_collect);
        [iter_pDC2_ave, iter_pDC2_min, iter_pDC2_max] = post_processing(iter_pDC2_collect);
        fprintf(fid,'pDCA fval (rho = 10): %.6f\t time: %.4f\t prob: %.4f\t iter: %.2f\n', fval_pDC2_ave, time_pDC2_ave, prob_pDC2_ave, iter_pDC2_ave);
    end
    if AL == 1
        [fval_ALDM_ave, fval_ALDM_min, fval_ALDM_max] = post_processing(fval_ALDM_collect);
        [prob_ALDM_ave, prob_ALDM_min, prob_ALDM_max] = post_processing(prob_ALDM_collect);
        [time_ALDM_ave, time_ALDM_min, time_ALDM_max] = post_processing(time_ALDM_collect);
        [iter_ALDM_ave, iter_ALDM_min, iter_ALDM_max] = post_processing(iter_ALDM_collect);
        fprintf(fid,'ALDM fval: %.6f\t time: %.4f\t prob: %.4f\t iter: %.2f\n', fval_ALDM_ave, time_ALDM_ave, prob_ALDM_ave, iter_ALDM_ave);
    end
    if SC == 1
        [fval_SCA_ave, fval_SCA_min, fval_SCA_max] = post_processing(fval_SCA_collect);
        [prob_SCA_ave, prob_SCA_min, prob_SCA_max] = post_processing(prob_SCA_collect);
        [time_SCA_ave, time_SCA_min, time_SCA_max] = post_processing(time_SCA_collect);
        [iter_SCA_ave, iter_SCA_min, iter_SCA_max] = post_processing(iter_SCA_collect);
        fprintf(fid,'SCA fval: %.6f\t time: %.4f\t prob: %.4f\t iter: %.2f\n', fval_SCA_ave, time_SCA_ave, prob_SCA_ave, iter_SCA_ave);
    end
    fclose(fid);
end
