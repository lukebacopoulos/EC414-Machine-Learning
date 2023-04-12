function [lambda_top5, k_] = skeleton_hw5_2()
%% Q5.2
%% Load AT&T Face dataset
    img_size = [112,92];   % image size (rows,columns)
    % Load the AT&T Face data set using load_faces()
    %%%%% TODO
    img_vector = load_faces();
    [n,d] = size(img_vector);

    %% Compute mean face and the covariance matrix of faces
    % compute X_tilde1
    %%%%% TODO
    X_tilde = zeros(n,d);
    meanface = mean(img_vector,1);
    for i = 1:n
        X_tilde(i,:) = img_vector(i,:) - meanface;
    end


    % Compute covariance matrix using X_tilde
    %%%%% TODO
    
    S_X = cov(X_tilde);

    %% Compute the eigenvalue decomposition of the covariance matrix
    %%%%% TODO
    
    [V,D] = eig(S_X);
    eigenvalues = diag(D);

    %% Sort the eigenvalues and their corresponding eigenvectors construct the U and Lambda matrices
    %%%%% TODO
    
   eigenvalues_sorted = flip(eigenvalues);
    Lamda = diag(eigenvalues_sorted);
    U = flip(V);

    
    %% Compute the principal components: Y
    %%%%% TODO

%% Q5.2 a) Visualize the loaded images and the mean face image
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image number 120 in the dataset
    % practice using subplots for later parts
    subplot(1,2,1)
    %%%%% TODO
    imshow(uint8(reshape(img_vector(120,:), img_size)));

    % Visualize the mean face image
    subplot(1,2,2)
    %%%%% TODO
    imshow(uint8(reshape(meanface, img_size)));

%% Q5.2 b) Analysing computed eigenvalues
    warning('off')
    
    % Report the top 5 eigenvalues
     lambda_top5 = eigenvalues_sorted(1:5,1); %%%%% TODO
     for i = 1:5
         fprintf('Lambda %d is %f\n',i,lambda_top5(i));
     end
    % Plot the eigenvalues in from largest to smallest
    k = 1:450;
    figure(2)
    sgtitle('Eigenvalues from largest to smallest')
    scatter(k,eigenvalues_sorted(k,1),'o','fill','b');

    % Plot the eigenvalue number k against k
    subplot(1,2,1)
    %%%%% TODO
    eigensum = sum(eigenvalues_sorted);
    partialsum = cumsum(eigenvalues_sorted);
    rho_k = partialsum / eigensum;
    rho_k = round(rho_k,2);

    % Plot the sum of top k eigenvalues, expressed as a fraction of the sum of all eigenvalues, against k
    %%%%% TODO: Compute eigen fractions
    
    subplot(1,2,2)
    scatter(k,rho_k(1:450,1),'o','fill','b');
    %%%%% TODO
    
    % find & report k for which the eigen fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    %%%%% TODO (Hint: ismember())
     k_ = zeros(length(ef),1); %%%%% TODO
     for i = 1:length(ef)
         ind = find(rho_k == ef(i));
        k_(i,1) = ind(1,1);
     end

%% Q5.2 c) Approximating an image using eigen faces
    test_img_idx = 43;
    test_img = img_vector(test_img_idx,:);    
    % Compute eigenface coefficients
    %%%% TODO
    K_array = [0,1,2,6,29,105,179,300,400];

    figure(3)
    sgtitle('Approximating original image by adding eigen faces')

    for i = 1:length(K_array)
        K = K_array(1,i);
    x_hat = PCAapprox(meanface,K,U,test_img);

    
    % add eigen faces weighted by eigen face coefficients to the mean face
    % for each K value
    % 0 corresponds to adding nothing to the mean face

    % visulize and plot in a single figure using subplots the resulating image approximations obtained by adding eigen faces to the mean face.

    %%%% TODO 
    subplot(3,3,i)
    imshow(uint8(reshape(x_hat, img_size)));
    end

%% Q5.2 d) Principal components capture different image characteristics
%% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of percentile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,~] = size(X);
    
    %% Compute the mean face and the covariance matrix
    % compute X_tilde
    %%%%% TODO
    
    % Compute covariance using X_tilde
    %%%%% TODO
    
    %% Compute the eigenvalue decomposition
    %%%%% TODO
    
    %% Sort the eigenvalues and their corresponding eigenvectors in the order of decreasing eigenvalues.
    %%%%% TODO
    
    %% Compute principal components
    %%%%% TODO
    
    %% Computing the first 2 pricipal components
    %%%%% TODO

    % finding percentile points
    percentile_vals = [5, 25, 50, 75, 95];
    %%%%% TODO (Hint: Use the provided fucntion - percentile_points())
    
    % Finding the cartesian product of percentile points to find grid corners
    %%%%% TODO

    
    %% Find images whose PCA coordinates are closest to the grid coordinates 
    
    %%%%% TODO

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 100th image
    subplot(1,2,1)
    %%%%% TODO
    
    % Mean face image
    subplot(1,2,2)
    %%%%% TODO

    
    %% Image projections onto principal components and their corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the principal component 1 vs principal component 2. Draw the
    % grid formed by the percentile points and highlight the image points that are closest to the 
    % percentile grid corners
    
    %%%%% TODO (hint: Use xticks and yticks)

    xlabel('Principal component 1')
    ylabel('Principal component 2')
    title('Image points closest to percentile grid corners')
    hold off
    
    figure(6)
    sgtitle('Images closest to percentile grid corners')
    hold on
    % Plot the images whose PCA coordinates are closest to the percentile grid 
    % corners. Use subplot to put all images in a single figure in a grid.
    
    %%%%% TODO
    
    hold off    
end



function xhat = PCAapprox(mu,k,U,X)
    y_cum = 0;

    if (k == 0)
        xhat = mu;
    else
        for i = 1:k
            y_pca = (X-mu) * U(:,i);
            y_cum = y_cum + y_pca;
        end
        xhat = mu + (y_cum * U(:,i)');
    end
end