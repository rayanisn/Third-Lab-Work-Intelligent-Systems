% Approximation using multilayer perceptron
clc
clear all
close all

% Weight and Biases
w = rand(2, 1);
w0 = rand(1, 1);

%INPUT
x = 0.1: 1/22: 1;

%OUTPUT
y = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2;

%Parameters for RBF
c1 = 0.2; r1 = 0.2475; %c1 is first peak on expected value
c2 = 0.9; r2 = 0.2475; %c2 is second peak on expected value
%r1 and r2 are result of r1 = abs(c2 - c1) / sqrt(2 * 2);

% Training parameters
eta = 0.1;                  % Training rate
max_epochs = 2000;          % Maximum number of training "rounds"
epoch = 0;                  % Initial epoch
error_threshold = 1e-9;     % Treshold to stop training
prev_avg_error = 1;         % Initialize previous error 

% Activation functions
RBF1 = exp(-((x - c1).^2) / (2 * r1^2)); 
RBF2 = exp(-((x - c2).^2) / (2 * r2^2)); 
RBF = [RBF1, RBF2];

while epoch < max_epochs
    e_total = 0; % Resets total error to 0 for each epoch
    output = zeros(size(x)); % Initialize the output vector 
    for i = 1:length(x)
        X = x(i);                  % Current input
        target = y(i);             % Target output for the current input

        % RBF layer
        RBF_output = [RBF1(i); RBF2(i)]; % RBF outputs for current input

        % Compute output layer input and output
        input = RBF_output' * w;         % Weighted sum 
        output(i) = input + w0;          % Add bias and store in output 

        % Calculate error
        error = target - output(i);      
        e_total = e_total + error^2;   

        % Update weights and bias
        w = w + eta * error * RBF_output;
        w0 = w0 + eta * error;
    end
    % Calculate average squared error 
    avg_error = e_total / length(x);
    
    % Check if average squared error < error threshold
    if abs(prev_avg_error - avg_error) < error_threshold
        break;
    end

    % Update previous average error and increment epoch
    prev_avg_error = avg_error;
    epoch = epoch + 1; 
    fprintf('Epoch %d, Average Error: %f, Error Change: %f\n', epoch, avg_error, abs(prev_avg_error - avg_error));
end


% Plot outputs
figure;
plot(x, y, 'b-', 'LineWidth', 1.5); 
hold on;
plot(x, output, 'r--', 'LineWidth', 1.5);
legend('Expected Output', 'Current Output');
xlabel('Input (x)');
ylabel('Output');
title('Expected vs. Current Output');
grid on;
hold off;







   
        