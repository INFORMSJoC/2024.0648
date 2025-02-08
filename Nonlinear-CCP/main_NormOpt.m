%%%%%%%%%%%%%%%%%%% Tests on Norm Optimization Problem %%%%%%%%%%%%%%%%%%%%%
% min -sum_i^n(x_i)
% s.t. P(sum_i^n(xi_ij^2 * x_i^2) <= U^2, j = 1,...,m) >= 1 - alpha,
%      x_i >= 0, i = 1,...,n.
clear; clc;
addpath(genpath(pwd));

fprintf('***************** Tests on Norm Optimization Problem *****************\n');
%% parameter setting
iid = 0;        % iid sample: 1
m = 20;         % number of constraints
n = 20;         % variable dimension
for N = [500, 1000, 1500, 2000] % sample size
    for alpha = [0.05, 0.1]     % risk level
        %% choose the method
        IP = 1; CVa = 1; Bi = 1; DC = 1; pDC = 1; pDC1 = 1; pDC2 = 1; SC = 1; repeat_num = 5;

        %% collect iteration information
        x_MIP_collect = {};
        x_CVa_collect = {};
        x_Bi_collect = {};
        x_DC_collect = {};
        x_PD_collect = {};
        x_PD_collect1 = {};
        x_PD_collect2 = {};
        x_SC_collect = {};

        [iter_Bi_collect, iter_PD_collect, iter_PD_collect1, iter_PD_collect2, iter_DC_collect, iter_SC_collect] = deal(zeros(repeat_num, 1));
        [fval_MIP_collect, fval_CVaR_collect, fval_DC_collect, fval_PD_collect, fval_PD_collect1, fval_PD_collect2, fval_Bi_collect, fval_SC_collect] = deal(zeros(repeat_num, 1));
        [prob_MIP_collect, prob_CVaR_collect, prob_DC_collect, prob_PD_collect, prob_PD_collect1, prob_PD_collect2, prob_Bi_collect, prob_SC_collect] = deal(zeros(repeat_num, 1));
        [time_MIP_collect, time_CVaR_collect, time_DC_collect, time_PD_collect, time_PD_collect1, time_PD_collect2, time_Bi_collect, time_SC_collect] = deal(zeros(repeat_num, 1));
        [ttime_MIP_collect, ttime_CVaR_collect, ttime_DC_collect, ttime_PD_collect, ttime_PD_collect1, ttime_PD_collect2, ttime_Bi_collect, ttime_SC_collect] = deal(zeros(repeat_num, 1));

        for iter = 1:repeat_num
            %% generate samples
            if iid == 1
                train_samples = zeros(N, n, m);
                for j = 1:m
                    train_samples(:, :, j) = normrnd(0, 1, N, n); % N x n x m
                end
            else
                train_samples = gensample(N, n, m);
            end

            %% parameter setting
            solver = 'gurobi'; maxitime = 1800; tol = 1e-4; U = 100;

            fprintf('***************** num of repeat: %d ***************** \n', iter);
            %% solve MIP by Gurobi
            if IP == 1
                opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol); upper = 1e4;
                MIP_t = tic; [x_MIP,time_MIP] = MIP(train_samples, U, upper, alpha, opts); ttime_MIP = toc(MIP_t);
                fval_MIP = -sum(x_MIP); prob_MIP = risk_level(train_samples, x_MIP, m);
                x_MIP_collect{iter} = x_MIP; fprintf('MIP %f\n', fval_MIP);
                fval_MIP_collect(iter) = fval_MIP;
                prob_MIP_collect(iter) = prob_MIP;
                ttime_MIP_collect(iter) = ttime_MIP; time_MIP_collect(iter) = time_MIP;
            end
            %% solve CVaR by Gurobi
            if CVa == 1
                opts = struct('solver', solver, 'maxitime', maxitime);
                CVa_t = tic; [x0,time_CVaR] = CVaR(train_samples, U, alpha, opts); ttime_CVaR = toc(CVa_t);
                fval_CVaR = -sum(x0); prob_CVaR = risk_level(train_samples, x0, m);
                x_CVa_collect{iter} = x0;
                fprintf('***************** CVaR ***************** \n');
                fprintf('CVaR %f\n', fval_CVaR);
                fval_CVaR_collect(iter) = fval_CVaR; prob_CVaR_collect(iter) = prob_CVaR;
                ttime_CVaR_collect(iter) = ttime_CVaR; time_CVaR_collect(iter) = time_CVaR;
            end
            %% Bisection Based CVaR
            if Bi == 1
                opts = struct('solver', solver, 'maxitime', maxitime);
                Bi_t = tic; [x_Bi, iter_Bi] = BiCVaR(train_samples, U, alpha, opts); ttime_Bi = toc(Bi_t);
                fval_Bi = -sum(x_Bi); prob_Bi = risk_level(train_samples, x_Bi, m);
                x_Bi_collect{iter} = x_Bi;
                iter_Bi_collect(iter) = iter_Bi; fprintf('BiCVaR %f\n', fval_Bi);
                fval_Bi_collect(iter) = fval_Bi; prob_Bi_collect(iter) = prob_Bi;
                ttime_Bi_collect(iter) = ttime_Bi;
            end
            %% DC algorithm
            if DC == 1
                solver = 'gurobi'; init = 1; %% init = 1: initialization by CVaR
                if init == 1
                    opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
                else
                    opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
                end
                DC_t = tic; [x_DC, time_DC, iter_DC] = DCA(train_samples, U, alpha, opts); ttime_DC = toc(DC_t);
                fval_DC = -sum(x_DC); prob_DC = risk_level(train_samples, x_DC, m);
                x_DC_collect{iter} = x_DC; fprintf('DCA %f\n', fval_DC);
                iter_DC_collect(iter) = iter_DC;
                fval_DC_collect(iter) = fval_DC; prob_DC_collect(iter) = prob_DC;
                ttime_DC_collect(iter) = ttime_DC; time_DC_collect(iter) = time_DC;
            end
            %% proximal DC algorithm
            if pDC == 1
                solver = 'gurobi'; beta = 0.1; init = 1; %% init = 1: initialization by CVaR
                if init == 1
                    opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
                else
                    opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
                end
                PD_t = tic; [x_PD, time_PD, iter_PD] = pDCA(train_samples, U, alpha, beta, opts); ttime_PD = toc(PD_t);
                fval_PD = -sum(x_PD); prob_PD = risk_level(train_samples, x_PD, m);
                x_PD_collect{iter} = x_PD; fprintf('pDCA (beta = 0.1) %f\n', fval_PD);
                iter_PD_collect(iter) = iter_PD;
                fval_PD_collect(iter) = fval_PD; prob_PD_collect(iter) = prob_PD;
                ttime_PD_collect(iter) = ttime_PD; time_PD_collect(iter) = time_PD;
            end
            if pDC1 == 1
                solver = 'gurobi'; beta = 1; init = 1; %% init = 1: initialization by CVaR
                if init == 1
                    opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
                else
                    opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
                end
                PD1_t = tic; [x_PD1, time_PD1, iter_PD1] = pDCA(train_samples, U, alpha, beta, opts); ttime_PD1 = toc(PD1_t);
                fval_PD1 = -sum(x_PD1); prob_PD1 = risk_level(train_samples, x_PD1, m);
                x_PD_collect1{iter} = x_PD1; fprintf('pDCA (beta = 1) %f\n', fval_PD1);
                iter_PD_collect1(iter) = iter_PD1;
                fval_PD_collect1(iter) = fval_PD1; prob_PD_collect1(iter) = prob_PD1;
                ttime_PD_collect1(iter) = ttime_PD1; time_PD_collect1(iter) = time_PD1;
            end
            if pDC2 == 1
                solver = 'gurobi'; beta = 10; init = 1; %% init = 1: initialization by CVaR
                if init == 1
                    opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
                else
                    opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
                end
                PD2_t = tic; [x_PD2, time_PD2, iter_PD2] = pDCA(train_samples, U, alpha, beta, opts); ttime_PD2 = toc(PD2_t);
                fval_PD2 = -sum(x_PD2); prob_PD2 = risk_level(train_samples, x_PD2, m);
                x_PD_collect2{iter} = x_PD2; fprintf('pDCA (beta = 10) %f\n', fval_PD2);
                iter_PD_collect2(iter) = iter_PD2;
                fval_PD_collect2(iter) = fval_PD2; prob_PD_collect2(iter) = prob_PD2;
                ttime_PD_collect2(iter) = ttime_PD2; time_PD_collect2(iter) = time_PD2;
            end
            %% Sequential Convex Approximation for DC Approximation
            if SC == 1
                solver = 'gurobi'; epsilon = 1; init = 1; %% init = 1: initialization by CVaR
                if init == 1
                    opts = struct('solver', solver, 'maxitime', maxitime, 'x0', x0, 'tol', tol);
                else
                    opts = struct('solver', solver, 'maxitime', maxitime, 'tol', tol);
                end
                SC_t = tic; [x_SC, time_SC, iter_SC] = SCA(train_samples, U, alpha, epsilon, opts); ttime_SC = toc(SC_t);
                x_SC_collect{iter} = x_SC;
                fval_SC = -sum(x_SC); prob_SC = risk_level(train_samples, x_SC, m);
                fprintf('SCA %f\n', fval_SC);
                iter_SC_collect(iter) = iter_SC;
                fval_SC_collect(iter) = fval_SC; prob_SC_collect(iter) = prob_SC;
                ttime_SC_collect(iter) = ttime_SC;  time_SC_collect(iter) = time_SC;
            end
        end
        %% print information
        fid = fopen(['Normopt_date',datestr(now,'mm-dd_HH-MM'),'_a=',num2str(alpha),'_N=',num2str(N),'_m=',num2str(m),'n=',num2str(n),'.txt'],'wt');
        fprintf('n = %d, m = %d, num of samples =%d\n', n, m, N);
        if IP == 1
            [fval_MIP_ave, fval_MIP_min, fval_MIP_max] = post_processing(fval_MIP_collect);
            [prob_MIP_ave, prob_MIP_min, prob_MIP_max] = post_processing(prob_MIP_collect);
            [time_MIP_ave, time_MIP_min, time_MIP_max] = post_processing(ttime_MIP_collect);
            fprintf(fid,'MIP\t %.4f\t %.4f\t %.4f\n', fval_MIP_ave, time_MIP_ave, prob_MIP_ave);
        end
        if CVa == 1
            [fval_CVa_ave, fval_CVa_min, fval_CVa_max] = post_processing(fval_CVaR_collect);
            [prob_CVa_ave, prob_CVa_min, prob_CVa_max] = post_processing(prob_CVaR_collect);
            [time_CVa_ave, time_CVa_min, time_CVa_max] = post_processing(ttime_CVaR_collect);
            fprintf(fid,'CVaR\t %.4f\t %.4f\t %.4f\n', fval_CVa_ave, time_CVa_ave, prob_CVa_ave);
        end
        if Bi == 1
            [fval_Bi_ave, fval_Bi_min, fval_Bi_max] = post_processing(fval_Bi_collect);
            [prob_Bi_ave, prob_Bi_min, prob_Bi_max] = post_processing(prob_Bi_collect);
            [time_Bi_ave, time_Bi_min, time_Bi_max] = post_processing(ttime_Bi_collect);
            [iter_Bi_ave, iter_Bi_min, iter_Bi_max] = post_processing(iter_Bi_collect);
            fprintf(fid,'BiCVaR\t %.4f\t %.4f\t %.4f\t %.2f\n', fval_Bi_ave, time_Bi_ave, prob_Bi_ave, iter_Bi_ave);
        end
        if DC == 1
            [fval_DC_ave, fval_DC_min, fval_DC_max] = post_processing(fval_DC_collect);
            [prob_DC_ave, prob_DC_min, prob_DC_max] = post_processing(prob_DC_collect);
            [time_DC_ave, time_DC_min, time_DC_max] = post_processing(ttime_DC_collect);
            [iter_DC_ave, iter_DC_min, iter_DC_max] = post_processing(iter_DC_collect);
            fprintf(fid,'DCA\t %.4f\t %.4f\t %.4f\t %.2f\n', fval_DC_ave, time_DC_ave, prob_DC_ave, iter_DC_ave);
        end
        if pDC == 1
            [fval_PD_ave, fval_PD_min, fval_PD_max] = post_processing(fval_PD_collect);
            [prob_PD_ave, prob_PD_min, prob_PD_max] = post_processing(prob_PD_collect);
            [time_PD_ave, time_PD_min, time_PD_max] = post_processing(ttime_PD_collect);
            [iter_PD_ave, iter_PD_min, iter_PD_max] = post_processing(iter_PD_collect);
            fprintf(fid,'pDCA(rho=0.1)\t %.4f\t %.4f\t %.4f\t %.2f\n', fval_PD_ave, time_PD_ave, prob_PD_ave,iter_PD_ave);
        end
        if pDC1 == 1
            [fval_PD1_ave, fval_PD1_min, fval_PD1_max] = post_processing(fval_PD_collect1);
            [prob_PD1_ave, prob_PD1_min, prob_PD1_max] = post_processing(prob_PD_collect1);
            [time_PD1_ave, time_PD1_min, time_PD1_max] = post_processing(ttime_PD_collect1);
            [iter_PD1_ave, iter_PD1_min, iter_PD1_max] = post_processing(iter_PD_collect1);
            fprintf(fid,'pDCA(rho=1)\t %.4f\t %.4f\t %.4f\t %.2f\n', fval_PD1_ave, time_PD1_ave, prob_PD1_ave, iter_PD1_ave);
        end
        if pDC2 == 1
            [fval_PD2_ave, fval_PD2_min, fval_PD2_max] = post_processing(fval_PD_collect2);
            [prob_PD2_ave, prob_PD2_min, prob_PD2_max] = post_processing(prob_PD_collect2);
            [time_PD2_ave, time_PD2_min, time_PD2_max] = post_processing(ttime_PD_collect2);
            [iter_PD2_ave, iter_PD2_min, iter_PD2_max] = post_processing(iter_PD_collect2);
            fprintf(fid,'pDCA(rho=10)\t %.4f\t %.4f\t %.4f\t %.2f\n', fval_PD2_ave, time_PD2_ave, prob_PD2_ave, iter_PD2_ave);
        end
        if SC == 1
            [fval_SC_ave, fval_SC_min, fval_SC_max] = post_processing(fval_SC_collect);
            [prob_SC_ave, prob_SC_min, prob_SC_max] = post_processing(prob_SC_collect);
            [time_SC_ave, time_SC_min, time_SC_max] = post_processing(ttime_SC_collect);
            [iter_SC_ave, iter_SC_min, iter_SC_max] = post_processing(iter_SC_collect);
            fprintf(fid,'SCA\t %.4f\t %.4f\t %.4f\t %.2f\n', fval_SC_ave, time_SC_ave, prob_SC_ave, iter_SC_ave);
        end
        fclose(fid);
        save(strcat('Normoptdate',datestr(now,'mm-dd_HH-MM'),'_N=',num2str(N)));
    end
end