% EC 414 - HW 3 - Spring 2022
% K-Means starter code

clear, clc, close all;

%% Generate Gaussian data:
% Add code below:
mu1 = [2,2];
mu2 = [-2,2];
mu3 = [0,-3.25];
sig1 = 0.02 * eye(2);
sig2 = 0.05 * eye(2);
sig3 = 0.07 * eye(2);
rng('default');
r1 = mvnrnd(mu1,sig1,50);
r2 = mvnrnd(mu2,sig2,50);
r3 = mvnrnd(mu3,sig3,50);
X = [r1;r2;r3];
figure
hold on
scatter(r1(:,1),r1(:,2),10,'.','red') % Scatter plot with points of size 10
scatter(r2(:,1),r2(:,2),10,'.','green') % Scatter plot with points of size 10
scatter(r3(:,1),r3(:,2),10,'.','blue') % Scatter plot with points of size 10

%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.

%% K-Means implementation
% Add code below

K = 3;
% MU_init = [3,3; -4,-1; 2,-4];
MU_init = [1,1; -2,0;0,-1];
%MU_init = [-0.14,2.61; 3.15,-0.84;-3.28,-1.58];
%  rng('default');
%  r = randi([-2,2]);
%  MU_init = r.* rand(3,2,'single');
MU_previous = MU_init;
MU_current = MU_init;


% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged == 0)
    MU_previous = MU_current;
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:

      [nextMU,labels] = clustering(X,MU_current);

      MU_current = nextMU;

    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    


    %% CODE 4 - Check for convergence 
    % Write code below here:
    MUdiff1 = sqrt((MU_current(1,1)-MU_previous(1,1)).^2+((MU_current(1,2)-MU_previous(1,2)).^2));
    MUdiff2 = sqrt((MU_current(2,1)-MU_previous(2,1)).^2+((MU_current(2,2)-MU_previous(2,2)).^2));
    MUdiff3 = sqrt((MU_current(3,1)-MU_previous(3,1)).^2+((MU_current(3,2)-MU_previous(3,2)).^2));
   
    
    MUavg = (MUdiff1 + MUdiff2 + MUdiff3) / 3;
   % scatter(MU_current(:,1),MU_current(:,2))
    if (MUavg <= convergence_threshold)
        converged=1;
    end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        hold on
        scatter(MU_current(:,1),MU_current(:,2))
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


