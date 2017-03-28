function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    fprintf('Size of theta')
    size(theta)
    fprintf('Size of sum((X*theta - y).*X))')
    size(sum((X*theta - y).*X))
    theta = theta - (alpha * 1/m * sum((X*theta - y).*X))'
    J_history(iter) = computeCost(X, y, theta);

end

end
