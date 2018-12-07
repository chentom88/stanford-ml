function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 - Compute the non-regularized cost function
% Feedforward to get outputs h(x) for each node for each sample 
a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Compute the cost
for i = 1:num_labels,
    % Get expected output
    y_label = (y == i);
    
    % h(x) for the output node corresponding to i
    hThetaX = a3(:, i);

    label_cost = sum( (-y_label .* log(hThetaX)) - ((1-y_label) .* log(1 - hThetaX)) );
    J = J + label_cost; 
end;

J = (1/m) * J;

% Part 2 - Compute Theta1_grad and Theta2_grad with backpropogation
delta2 = 0;
delta1 = 0;

for i = 1:m,
    % get forward prop activations for sample 
    z1_bp = a1(i, :)';

    z2_bp = [1 z2(i, :)]';
    a2_bp = a2(i, :)';

    a3_bp = a3(i, :)';

    % make y vector
    y_bp = zeros(num_labels, 1);
    y_bp(y(i)) = 1;

    % compute deltas
    d3 = (a3_bp - y_bp);
    d2 = Theta2'*d3 .* sigmoidGradient(z2_bp);
    d2 = d2(2:end);

    d1 = Theta1'*d2 .* sigmoidGradient(z1_bp);

    delta2 = delta2 + (d3 * a2_bp');
    delta1 = delta1 + (d2 * z1_bp');
end;

Theta2_grad = (1/m)*delta2;
Theta1_grad = (1/m)*delta1;

% Part 3 - Compute the regularized cost function
% remember not to include the bias column in either Theta
reg = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
J = J + reg;

theta2_reg = [zeros(num_labels, 1) Theta2(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda/m)*theta2_reg;

theta1_reg = [zeros(hidden_layer_size, 1) Theta1(:, 2:end)];
Theta1_grad = Theta1_grad + (lambda/m)*theta1_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
