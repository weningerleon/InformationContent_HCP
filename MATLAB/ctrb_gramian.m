function G = ctrb_gramian(A, B, T, c)
% Function for computing average controllability for each node.
% Unsure of numerical stability, don't use for publication results
%
% Inputs:
% A:    N x N continuous-time stable adjacency matrix
% B:    N x N input nodes
% T:    1 x 1 scalar for time horizon
% 
% Outputs:
% v:    N x 1 vector of average controllability values

% Normalize
A = A/(eigs(A,1)*(1+c)) - eye(size(A));


fun = @(t) expm(A*t)*B*B'*expm(A'*t);
G = integral(fun, 0, T, 'ArrayValued', 1,'AbsTol', 1e-12, 'RelTol', 1e-12);

end