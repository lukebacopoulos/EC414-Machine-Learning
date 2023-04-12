function [nextMU,y,wcdist] = clustering(X,currentMU)
[m,~] = size(X);
[K,~] = size(currentMU);
MU = currentMU;

distance = zeros(m,K);
% for j = 1:K
%     for i = 1:150
%         distance(i,j) = norm(X(i,:)-MU(j,:));
%     end
% end

  for i = 1:K
        distance(:,i) = sqrt(sum((X - MU(i,:)).^2, 2));
  end

[wcdist, y_hat] = min(distance, [], 2);

for i = 1:K
        index = find(y_hat == i);
        MU(i, :) = mean(X(index, :));
end

nextMU = MU;
y = y_hat;