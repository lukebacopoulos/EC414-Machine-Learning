%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2022
% HW 4
% Luke Bacopoulos baco@bu.edu


% FOR SUBPROBLEMS WITH MULTIPLE PARTS, PLEASE UNCOMMENT THE CORRESPONDING 
% CODE FOR THE PART.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
n = n1+n2;
mu1 = [1; 2];
mu2 = [3; 2];



%***********************  Uncomment for each part *************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate dataset (i) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% lambda1 = 1;
% lambda2 = 0.25;
% theta = 0;
% [X, X1, X2, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate dataset (ii) ************USE FOR REST OF PROBLEM***************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda1 = 1;
lambda2 = 0.25;
theta = pi/6;
[X, X1, X2, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate dataset (iii) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% lambda1 = 1;
% lambda2 = 0.25;
% theta = pi/3;
% [X, X1, X2, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate dataset (iv) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% lambda1 = 0.25;
% lambda2 = 1;
% theta = pi/6;
% [X, X1, X2, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset

figure(1);subplot(2,2,1);
scatter(X1(:,1),X1(:,2),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(0),'\times \pi/6']);
scatter(X2(:,1),X2(:,2),'^','fill','r');
axis equal;
hold off;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
%phi_array = pi/6;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = mean(mean(signal_power));
    noise_power_array(i) = mean(mean(noise_power));
    snr_array(i) = mean(mean(snr));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
hold on;
scatter(phi_array,signal_power_array,'o','fill','b');
scatter(phi_array,noise_power_array,'^','fill','r');
scatter(phi_array,snr_array,'s','fill','g');
xlabel('phi');
legend('Signal Power','Noise Power','SNR');

signal_sort = sort(signal_power_array);
noise_sort = sort(noise_power_array);
snr_sort = sort(snr_array);

signal_max = signal_sort(1,length(signal_sort));

    
noise_min = noise_sort(1,1);


snr_max = snr_sort(1,length(snr_sort));


signal_ind = find(signal_power_array == signal_max);
noise_ind = find(noise_power_array == noise_min);
snr_ind = find(snr_array == snr_max);

fprintf('Signal Maximum at ϕ = %f\n',phi_array(1,signal_ind));
fprintf('Noise Minimum at ϕ = %f\n',phi_array(1,noise_ind));
fprintf('SNR Maximum at ϕ = %f\n',phi_array(1,snr_ind));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...



%***********************  Uncomment for each part *************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% phi = 0;
% [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi, true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% phi = pi/6;
% [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi, true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% phi = pi/3;
% [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi, true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);

% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
muX1 = mean(X1);
muX2 = mean(X2);
phi = phi_array(1,snr_ind);
w_snr = [cos(phi);sin(phi)];
figure
hold on
scatter(X1(:,1),X1(:,2),'x','r');
scatter(X2(:,1),X2(:,2),'^','g');
scatter(muX1(:,1),muX1(:,2),'o','fill','b');
scatter(muX2(:,1),muX2(:,2),'o','fill','p');
quiver(muX1(1,1),muX1(1,2),w_LDA(1,1),w_LDA(2,1));
quiver(muX1(1,1),muX1(1,2),muX2(1,1),muX2(1,2));
quiver(muX1(1,1),muX1(1,2),-2 * w_snr(1,1),-2 * w_snr(2,1));
legend('Class 1','Class 2','Mu_X_1','Mu_X_2','w_L_D_A','Mu_X_2 - Mu_X_1','w_S_N_R');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot

X_project = X * w_LDA;
X_project_sorted = sort(X_project);
b_array = (diag(ones(1,n))+ diag(ones(1,n-1),-1)) * X_project_sorted / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
scatter(b_array(:,1),ccr_array(1,:),'o','fill','b');
xlabel('b value')
ylabel('CCR')
title('Correct Classification Rate Versus b Value')

ccr_ind = find(ccr_array == max(ccr_array));
b_max = b_array(ccr_ind);
fprintf('CCR is maximized when b = %f\n',b_max);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, X1, X2, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2


%%%%%%%%%%%%%%%%%%%%%%
%Insert your code here
%%%%%%%%%%%%%%%%%%%%%%

y1 = zeros(n1,1);
y2 = zeros(n2,1);
u1 = [cos(theta),sin(theta)];
u2 = [sin(theta),-cos(theta)];

Lamda = [lambda1,0;0,lambda2];
S = [u1;u2];
S_inv = inv(S);
sigma = S * Lamda * S_inv;

X1 = mvnrnd(mu1,sigma,n1);
X2 = mvnrnd(mu2,sigma,n2);

y1 = y1 + 1;
y2 = y2 + 2;

X = [X1;X2];
Y = [y1;y2];
end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = [cos(phi);sin(phi)];

X1 = X((find(Y == 1)),:);
X2 = X((find(Y == 2)),:);

n1 = length(X1);
n2 = length(X2);
n = n1+n2;
muX1 = mean(X1);
muX2 = mean(X2);

signal = ((muX2-muX1)*w)^2;

p1 = n1/n;
p2 = n2/n;

SX1 = cov(X1);
SX2 = cov(X2);

SXavg = (p1*SX1+p2*SX2);

noise = w' * SXavg * w;

snr = signal ./ noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity(X1);
    plot(pdf1,z1)
    hold on;
    [pdf2,z2] = ksdensity(X2);
    plot(pdf2,z2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title('Estimated class density estimates of data projected along \phi = ? \times \pi/6. Ground-truth \phi = \pi/6')
end

end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X1 = X((find (Y == 1)),:);
X2 = X((find (Y == 2)),:);

muX1 = mean(X1);
muX2 = mean(X2);

n1 = length(X1);
n2 = length(X2);
n = n1+n2;

SX1 = cov(X1);
SX2 = cov(X2);

SXavg = (n1/n)*SX1+(n2/n)*SX2;
SXavg_inv = inv(SXavg);
w_LDA = SXavg_inv*(muX2-muX1)';




end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(X);
y_pred = zeros(length(X),1);

for i = 1:length(X)
    h = X(i,:) * w_LDA + b;
    if h <= 0
        y_pred(i,1) = 1;
    else
        y_pred(i,1) = 2;
    end
end
correctclass = Y - y_pred;

correct = find(correctclass == 0);
numc = length(correct);
ccr = numc / n;
end