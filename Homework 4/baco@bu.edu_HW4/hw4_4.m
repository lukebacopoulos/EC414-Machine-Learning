
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   Luke Bacopoulos baco@bu.edu
%
%   Implementation of Ridge Regression
%   
%   HW4.4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
load prostateStnd.mat

% part (a)
% normalize training data
[n,d] = size(Xtrain);
one = ones(n,1);

% mean centered feature matrix:

mu_Xtrain = mean(Xtrain);
Xtrain_t = Xtrain - one * mu_Xtrain;

% mean centered label matrix:

mu_ytrain = mean(ytrain);
ytrain_t = ytrain - one * mu_ytrain;

%[w_OLS, b_OLS] = OLS(Xtrain_t,ytrain_t,mu_Xtrain,mu_ytrain)




for i = -5:1:10
   lamda = exp(i)
    [w_ridge, b_ridge] = ridgereg(Xtrain_t, ytrain_t, mu_Xtrain, mu_ytrain, lamda);
    ridgevec = zeros(15,2);
    ridgevec(i,1) = w_ridge; ridgevec(i,2) = b_ridge;
   



end














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%**************************   FUNCTIONS     *******************************

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w_OLS, b_OLS] = OLS(X_t,Y_t,Mu_x,Mu_y)

n = length(X_t);

S_x = 1/ n * (X_t' * X_t);
S_xinv = inv(S_x);

S_xy = 1 / n * X_t' * Y_t;


w_OLS = S_xinv * S_xy;

b_OLS = Mu_y - Mu_x * w_OLS;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [w_ridge, b_ridge] = ridgereg(X_t,Y_t,Mu_X,Mu_y,lamda)

[n, d] = size(X_t);
I = eye(d);

S_x = 1/ n * (X_t' * X_t);

S_xy = 1 / n * X_t' * Y_t;

w_int = inv(((lamda / n) * I) + S_x);

w_ridge = w_int * S_xy;

b_ridge = Mu_y - w_ridge' * Mu_X';


end



















