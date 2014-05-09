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
identity = eye(num_labels); %genero matrice identità per convertire y che ha valori da 1 a 10 in una matrice che abbia
                           %per ogni riga un vettore di zeri ed un 1 nella
                           %posizione indicata dal numero (1:10)
new_y = zeros(size(y,1), num_labels);
%new_y è la matrice convertita new_y = <5000X10>
%ad esempio se y(100) era 5 allora new_y(100,:)=[0 0 0 0 1 0 0 0 0 0]
for i=1:size(new_y,1)
    new_y(i,:)=identity(y(i),:);
end
     
X = [ones(m, 1) X];

z_2 = Theta1*X'; % <25X5000> = <25X401>*<401X5000>
a_2 = sigmoid(z_2); % <25X5000>
a_2 = [ones(1, size(a_2,2)); a_2]; % <26X5000>
z_3 = Theta2*a_2; % <10X5000> = <10X26>*<26X5000>
a_3 = sigmoid(z_3); % <10X5000>

for i=1:size(X,1)
    J=J+sum(new_y(i,:)*log(a_3(:,i)) + (1-new_y(i,:))*log(1-a_3(:,i)));
end

%J=(log(output)*new_y + log(1-output)*(1-new_y));
J=(-1/m)*J;

%adding regularization
squared1 = Theta1(:,2:end).^2;
squared2 = Theta2(:,2:end).^2;
reg=(lambda/(2*m))*(sum(squared1(:))+sum(squared2(:)));

J=J+reg;


% -------------------------------------------------------------
%computing gradient
DELTA_1 = zeros(size(Theta1));
DELTA_2 = zeros(size(Theta2));
for i=1:size(X,1)
    %Perform a feedforward pass computing the activations (z(2); a(2); z(3); a(3)) for layers 2 and 3.
    a_1i = X(i,:)'; % <401X1>
    z_2i = z_2(:,i); % <25X1>
    a_2i = a_2(:,i); % <26X1>
    z_3i = z_3(:,i); % <10X1>
    a_3i = a_3(:,i); % <10X1>
    y_i = new_y(i,:)'; % <10X1>
    
    %For each output unit k in layer 3 (the output layer), set_k(3) = (a_k(3) - y_k);
    delta_3i = a_3i - y_i; % <10X1>
    
    %third passage
    delta_2i = (Theta2'*delta_3i).*sigmoidGradient([1; z_2i]); % <26X1> = <26X10>*<10X1>.*<26X1>
    
    %accumulate the gradient
    DELTA_1 = DELTA_1 + delta_2i(2:end)*a_1i'; % prodotto estterno tra vettori <25X401> = <25X1>*<1X401>
    DELTA_2 = DELTA_2 + delta_3i*a_2i'; % prodotto estterno tra vettori <10X26> = <10X1>*<1X26>
end

Theta1(:,1) = zeros(size(Theta1,1), 1); 
Theta2(:,1) = zeros(size(Theta2,1), 1);

Theta1_grad = (1/m)*DELTA_1 + (lambda/m)*Theta1;
Theta2_grad = (1/m)*DELTA_2 + (lambda/m)*Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
