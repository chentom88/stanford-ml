function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
lowest_err = 1000;

for i = 1:size(C_range, 2),
    current_C = C_range(i);

    for j = 1:size(sigma_range, 2),
        current_sigma = sigma_range(j);

        model = svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
        pred = svmPredict(model, Xval);
        curr_err = mean(double(pred ~= yval));

        if (curr_err < lowest_err),
            lowest_err = curr_err;

            C = current_C;
            sigma = current_sigma;
        end;
    end;
end;

% =========================================================================

end
