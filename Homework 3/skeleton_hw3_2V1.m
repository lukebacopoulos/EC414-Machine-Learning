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
MU_init = [3,3; -4,-1; 2,-4];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0 && iteration <= 1000)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

   [nextMU,labels] = clustering(X,MU_current);

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
%     for i = 1:150
%         eucdist1(i,1) = norm(X(i,1)-MU_current(1,:)); % compute distances to Mu1
%         eucdist2(i,1) = norm(X(i,1)-MU_current(2,:)); % " " Mu2
%         eucdist3(i,1) = norm(X(i,1)-MU_current(3,:)); % " " Mu3
%         distances(i,1:3) = [eucdist1(i,1),eucdist2(i,1),eucdist3(i,1)]; % distance of each point to each Mu
%     end
% 
%     for i = 1:150
%     [sortedvals,ind] = sort(distances(i,:));     % sort values, label each point to smallest Mu
%     labels(i,1) = ind(1,1); 
%     end

    


    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
% cluster1 = zeros(1,2);
% ccount1 = 0;
% cluster2 = zeros(1,2);       % initialize clusters
% ccount2 = 0;
% cluster3 = zeros(1,2);
% ccount3 = 0;

MU_previous = MU_current;

for i = 1:K
    ind = find(labels == i);
    if (isempty(ind))
        MU_current(i,:) = randi([-3,3],1,2);
    
    else 
        MU_current(i,:) = mean(X(ind,:));
    end
end


%     if labels(i,1) == 1
%         cluster1x(i,1) = X(i,1);
%         cluster1y(i,1) = X(i,2);
%     elseif labels(i,1) == 2
%         cluster2x(i,1) = X(i,1);
%         cluster2y(i) = X(i,2);
%     elseif labels(i,1) == 3
%         cluster3x(i,1) = X(i,1);
%         cluster1y(i,1) = X(i,2);
%     end



% sumX1 = 0;
% sumY1 = 0;
% sumX2 = 0;
% sumY2 = 0;
% sumX3 = 0;
% sumY3 = 0;
% count1 = 0;
% count2 = 0;
% count3 = 0;
% 
%     for i = 1:length(labels)
%         l = labels(i,1);
%         if l == 1
%             sumX1 = sumX1 + X(i,1);
%             sumY1 = sumY1 + X(i,2);
%             count1 = count1 + 1;
%         elseif l == 2
%             sumX2 = sumX2 + X(i,1);
%             sumY2 = sumY2 + X(i,2);
%             count2 = count2 + 1;
%         elseif l == 3
%             sumX3 = sumX3 + X(i,1);
%             sumY3 = sumY3 + X(i,2);
%             count3 = count3 + 1;
%         end
%     end
% 
%     Mu1 = [sumX1 ./ count1, sumY1 ./ count1];
%     Mu2 = [sumX2 ./ count2, sumY2 ./ count2];
%     Mu3 = [sumX3 ./ count3, sumY3 ./ count3];
% 
%     nextMu = [Mu1;Mu2;Mu3];
%     if count1 == 0
%       nextMu(1,:) = [2,2]; 
%     elseif count2 == 0
%         nextMu(2,:) = [-2,2];
%     elseif count3 == 0 
%        nextMu(3,:) = [0,-2];
%     end


    %% CODE 4 - Check for convergence 
    % Write code below here:
    MUdiff1 = (MU_current(1,1)-MU_previous(1,1)).^2+((MU_current(1,2)-MU_previous(1,2)).^2);
    MUdiff2 = (MU_current(2,1)-MU_previous(2,1)).^2+((MU_current(2,2)-MU_previous(2,2)).^2);
    MUdiff3 = (MU_current(3,1)-MU_previous(3,1)).^2+((MU_current(3,2)-MU_previous(3,2)).^2);
   
    
    MUavg = (MUdiff1 + MUdiff2 + MUdiff3) / 3;
   % scatter(MU_current(:,1),MU_current(:,2))
    if (MUavg <= convergence_threshold)
        converged=1;
    end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1 || iteration >= 999)
        fprintf('\nConverged.\n')
        hold on
        scatter(MU_current(:,1),MU_current(:,2))
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


