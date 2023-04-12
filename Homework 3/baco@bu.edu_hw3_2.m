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
% figure
% hold on
% scatter(r1(:,1),r1(:,2),10,'.','red') % Scatter plot with points of size 10
% scatter(r2(:,1),r2(:,2),10,'.','green') % Scatter plot with points of size 10
% scatter(r3(:,1),r3(:,2),10,'.','blue') % Scatter plot with points of size 10

%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here

% Problem 3.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 3 folder on Blackboard.

%% K-Means implementation
% Add code below

K = 3;
J = 10;
WCSS_current = 0;
WCSSind = 1;
plotMU = zeros(K,2);
%MU_init = [3,3; -4,-1; 2,-4];
%MU_init = [-0.14,2.61; 3.15,-0.84;-3.28,-1.58];
%  rng('default');
%MUmat = randi([-2,2],K*J,2); 

% for i = 0:K:K*J-K
%     for j = 1:K
%         index = i + j;
%     end
%     MU_init(index,:) = MUmat(index,:);
% end

for i = 1:J
%    MU_init = MUmat(i*K-2:i*K,:);
MU_init = randi([-2,2],K,2);
MU_current = MU_init;


% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged ~= 1 && iteration < 10)
    MU_previous = MU_current;
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:

      [nextMU,labels,wcdist] = clustering(X,MU_current);

      MU_current = nextMU;

    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    


    %% CODE 4 - Check for convergence 
    % Write code below here:
    MUdiff1 = ((MU_current(1,1)-MU_previous(1,1)).^2+((MU_current(1,2)-MU_previous(1,2)).^2));
    MUdiff2 = ((MU_current(2,1)-MU_previous(2,1)).^2+((MU_current(2,2)-MU_previous(2,2)).^2));
    MUdiff3 = ((MU_current(3,1)-MU_previous(3,1)).^2+((MU_current(3,2)-MU_previous(3,2)).^2));
   
    
    MUavg = (MUdiff1 + MUdiff2 + MUdiff3) / 3;
   % scatter(MU_current(:,1),MU_current(:,2))
    if (MUavg <= convergence_threshold)
        converged=1;
    end
end
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
%         figure
%         hold on
% 
%             index = find(labels == 1);
%         scatter(X(index,1),X(index,2),10,'.','red') % Scatter plot with points of size 10
%             index = find(labels == 2);
%         scatter(X(index,1),X(index,2),10,'.','green') % Scatter plot with points of size 10
%             index = find(labels == 3);
%         scatter(X(index,1),X(index,2),10,'.','blue') % Scatter plot with points of size 10
%         
%         scatter(MU_current(:,1),MU_current(:,2))
        
        
        %% If converged, get WCSS metric
        % Add code below
        
    x = WCSS(wcdist,labels,MU_current);

    if (WCSS_current == 0)
        WCSS_current = x;
    elseif (x < WCSS_current)
        WCSS_current = x;
        WCSSind = J;
        plotMU= MU_current;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
fprintf('Best WCSS: %d\nfor MU %d\n',WCSS_current,WCSSind)
[~,labels,~] = clustering(X,plotMU);
figure
        hold on

            index = find(labels == 1);
        scatter(X(index,1),X(index,2),10,'.','red') % Scatter plot with points of size 10
            index = find(labels == 2);
        scatter(X(index,1),X(index,2),10,'.','green') % Scatter plot with points of size 10
            index = find(labels == 3);
        scatter(X(index,1),X(index,2),10,'.','blue') % Scatter plot with points of size 10
        
        scatter(plotMU(:,1),plotMU(:,2))
