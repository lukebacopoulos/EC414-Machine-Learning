function x = WCSS(dist,labels,MU)

[K,~] = size(MU);
out = 0;
for i = 1:K
    index = find(labels == i);
    wcsquares = norm(dist(index,:)-MU(i,:)).^2;
    out = out + sum(wcsquares);
end

x = out;
end