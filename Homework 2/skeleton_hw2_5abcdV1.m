% EC 414 Introduction to Machine Learning
% Spring 2022
% Homework 2
% by (fill in name)
%
% Nearest Neighbor Classifier
%
% Problem 2.5 a, b, c, d


clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()

% label axis and include title

gscatter(Xtrain(1:200,1),Xtrain(1:200,2),ytrain(1:200))
 xlabel('Xtrain1')
 ylabel('Xtrain2')
 title('Training Data')


% b)Plotting Probabilities on a 2D map
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

        k = 10;                                         % 10 nearest neighbors
        kNNmat = zeros(k,Ntest);
        distvec1 = zeros(Ntest,1);
        distvec2 = zeros(Ntest,1);
        totaldist = zeros(Ntest,1);
for i = 1:Ntest
        distvec1 = (Xtest(i,1)-Xtrain(:,1));           % dist between X and Xi
        distvec2 = (Xtest(i,2)-Xtrain(:,2));           % dist between Y and Yi
        totaldist = (distvec1.^2+distvec2.^2).^(1/2);   % Euclidean Distance

        [sortedvals,ind] = sort(totaldist);         % Find the smallest distances and their indices
                                           
        kNN = zeros(k,2);                           % Indicies of the 10 closest points
        kNN(:,1) = ind(1:k,1);                      % 

        for j = kNN(:,1)                            %
            kNN(:,2) = ytrain(j,1);                 % list the groups each point belongs to
        end
    kNNmat(:,i) = kNN(:,2);                         % Matrix of each point's ten nearest neighbors
end

% compute probabilities of being in class 2 for each point on grid

probabilities = zeros(Ntest,1);
for j = 1:Ntest
vec = kNNmat(:,j);
count = 0;                                          % initialize count variable outside of loop

for i = 1:length(vec)
    if vec(i,1) == 2
        count = count + 1;
    end
    prob = count / k;
end


probabilities(j,1) = prob;
end

% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('X1')
ylabel('X2')
title('Probability of Being Classified as Group 2')


% repeat steps above for class 3 below

class3probability = zeros(Ntest,1);
for i = 1:Ntest
    vec3 = kNNmat(:,i);
    count = 0;

for j = 1:length(vec3)
    if vec3(j,1) == 3
        count = count + 1;
    end
    prob3 = count / k;
end
class3probability(i,1) = prob3;
end

% Figure for class 3
figure
class3ProbonGrid = reshape(class3probability,size(Xgrid));
contourf(Xgrid,Ygrid,class3ProbonGrid);
colorbar;
% remember to include title and labels!
xlabel('X1')
ylabel('X2')
title('Probability of Being Classified as Group 3')

% c) Class label predictions
K = 1 ; % K = 1 case
                                         % 10 nearest neighbors
        kNNmat = zeros(K,Ntest);
        distvec1 = zeros(Ntest,1);
        distvec2 = zeros(Ntest,1);
        totaldist = zeros(Ntest,1);
for i = 1:Ntest
        distvec1 = (Xtest(i,1)-Xtrain(:,1));           % dist between X and Xi
        distvec2 = (Xtest(i,2)-Xtrain(:,2));           % dist between Y and Yi
        totaldist = (distvec1.^2+distvec2.^2).^(1/2);   % Euclidean Distance

        [sortedvals,ind] = sort(totaldist);         % Find the smallest distances and their indices
                                           
        kNN = zeros(K,2);                           % Indicies of the 10 closest points
        kNN(:,1) = ind(1:K,1);                      % 

        for j = kNN(:,1)                            %
            kNN(:,2) = ytrain(j,1);                 % list the groups each point belongs to
        end
    kNNmat(:,i) = kNN(:,2);                         % Matrix of each point's ten nearest neighbors
end
% compute predictions 
ypred = kNNmat;
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
%remember to include title and labels!
xlabel('1 NN Predictions')
ylabel('X2')
title('X1')

% repeat steps above for the K=5 case. Include code for this below.
K = 5;
kNNmat = zeros(K,Ntest);
        distvec1 = zeros(Ntest,1);
        distvec2 = zeros(Ntest,1);
        totaldist = zeros(Ntest,1);
for i = 1:Ntest
        distvec1 = (Xtest(i,1)-Xtrain(:,1));           % dist between X and Xi
        distvec2 = (Xtest(i,2)-Xtrain(:,2));           % dist between Y and Yi
        totaldist = (distvec1.^2+distvec2.^2).^(1/2);   % Euclidean Distance

        [sortedvals,ind] = sort(totaldist);         % Find the smallest distances and their indices
                                           
        kNN = zeros(K,2);                           % Indicies of the 10 closest points
        kNN(:,1) = ind(1:K,1);                      % 

        for j = kNN(:,1)                            %
            kNN(:,2) = ytrain(j,1);                 % list the groups each point belongs to
        end
    kNNmat(:,i) = kNN(:,2);                         % Matrix of each point's ten nearest neighbors
end

k5pred = zeros(Ntest,1);

for j = 1:Ntest
vec5 = kNNmat(:,j);
count1 = 0;                                          % initialize count variable outside of loop
count2 = 0;
count3 = 0;
for i = 1:length(vec5)
    if vec5(i,1) == 1
        count1 = count1 + 1;
    elseif vec5(i,1) == 2
        count2 = count2 + 1;
    elseif vec5(i,1) == 3
        count3 = count3 + 1;
    end
end
    if count3 >= count2 && count3 >= count1
        k5pred(j,1) = 3;
    elseif count2 >= count1
        k5pred(j,1) = 2;
    else
        k5pred(j,1) = 1;
    end
    
end

ypred = k5pred;
figure
gscatter(Xgrid(:),Ygrid(:),ypred,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
%remember to include title and labels!
xlabel('5 NN Predictions')
ylabel('X2')
title('X1')


% d) LOOCV CCR computations
Ntest = 200;
for K = 1:2:11
    % determine leave-one-out predictions for k
    kNNmat = zeros(K,Ntest);
        distvec1 = zeros(Ntest,1);
        distvec2 = zeros(Ntest,1);
        totaldist = zeros(Ntest,1);
for i = 1:Ntest
        distvec1 = (Xtrain(i,1)-Xtrain(:,1));           % dist between X and Xi
        distvec2 = (Xtrain(i,2)-Xtrain(:,2));           % dist between Y and Yi
        totaldist = (distvec1.^2+distvec2.^2).^(1/2);   % Euclidean Distance

        [sortedvals,ind] = sort(totaldist);         % Find the smallest distances and their indices
                                           
        kNN = zeros(K,2);                           % Indicies of the 10 closest points
        kNN(:,1) = ind(1:K,1);                      % 

        for j = kNN(:,1)                            %
            kNN(:,2) = ytrain(j,1);                 % list the groups each point belongs to
        end
    kNNmat(:,i) = kNN(:,2);                         % Matrix of each point's ten nearest neighbors
end
    
    
    
  %  ypred = 


    % compute confusion matrix
    conf_mat = confusionmat(Ygrid(:), ypred);
    % from confusion matrix, compute CCR
 %   CCR = 
    
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end
end

% % plot CCR values for k = 1,3,5,7,9,11
% % label x/y axes and include title
