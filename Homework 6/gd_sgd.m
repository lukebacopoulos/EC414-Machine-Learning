clear;clc;close

% f(x)=||x-[1;1]||^2+||x-[3;5]||^2

x = [-1;0]
eta = 0.1;

X = x;

for i = 1:500
    g1 = 2 * (x - [1;1]);
    g2 = 2 * (x - [3;5]);
    
    %GD
%     g = g1 + g2;
    
    %SGD
%     if mod(i,2) == 1
    if rand() < 0.5
        g = g1;
    else
        g = g2;
    end
    
    x = x - eta * g;
    X = [X,x];
end
X

plot(X(1,:),X(2,:),'s -')
hold on
plot(2,3,'or')