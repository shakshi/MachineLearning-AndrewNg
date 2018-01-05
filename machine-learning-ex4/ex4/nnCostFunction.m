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

%To compute h(x)
a_1=  X';
a_1= [ ones(1,m) ; a_1];  %adding bias for all egs

z_2= Theta1 * a_1;
a_2= sigmoid(z_2);

a_2= [ ones(1,m) ; a_2];  %adding bias for all egs
z_3= Theta2 * a_2;
a_3= sigmoid(z_3);
h=a_3;
% a_3= [ ones(1,m) ; a_3];  % as a_3 is to be used later, adding bias for all egs

% To compute new y
y2=zeros(num_labels,m);
for i=1:m,    %for i columns
  y2(y(i),i)=1;
end

% To compute cost 
cost= (log(h) .* y2) + (log(1-h) .* (1-y2));
total_cost= sum(cost(:));
sum_cost= -1/m * total_cost;

% to compute regularization cost
theta1_reg= Theta1; theta1_reg(:,1)=0;
theta2_reg= Theta2; theta2_reg(:,1)=0;

p= theta1_reg .^ 2;
q= theta2_reg .^ 2;
sum_sq= sum(p(:)) + sum (q(:));

J= (-1/m * total_cost) + ( lambda/ (2*m) * sum_sq ) ;

%% Back Propagation 

% S_3 and S_2

y3= [ones(1,m) ; y2];    % as bias units in any layer will always be 1.. therefore desired o/p is 1
S_3= a_3 - y2;
t=Theta2(:,2:size(Theta2,2));
u=a_2(2: size(a_2,1), :);
S_2= (t'* S_3) .* u .* (1-u);

% delta_2 and delta_1

delta_2= S_3 * (a_2');
delta_1= S_2 * (a_1');

% d_2 and d_1

d_2= (1/m) * (delta_2 + (lambda * theta2_reg));
d_1= (1/m) * (delta_1 + (lambda * theta1_reg));

% Finally 
Theta2_grad = d_2;
Theta1_grad = d_1;

% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
