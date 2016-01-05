function prob = algorithm(q)

load sp500.mat
%% Initializations
% pi: Initial probabilities
% A: Transition probabilities
% B: Emission probabilities

pi = [0.2; 0.8];
A = [0.8, 0.2; 0.2, 0.8];
B = [q, 1-q; 1-q, q];

alpha = zeros(2, numel(price_move));
beta = ones(2, numel(price_move));

pi_new = zeros(2,1);
A_new = zeros(2,2);
B_new = zeros(2,2);

%% Algorithm

% Forward Algorithm
% % Matrix method
alpha(:,1) = pi.*B(:,obs2ind(price_move(1)));
s = zeros(2,numel(price_move));
s(1) = 1;
for i = 2:numel(price_move)
    alpha(:,i) = B(:,obs2ind(price_move(i))).*(A'*alpha(:,i-1));
%     s(i) =  sum(alpha(:,i));
%     alpha(:,i) = alpha(:,i)./s(i);
end

% Loop method
% alpha(:,1) = pi.*B(:,obs2ind(price_move(1)));
% s = zeros(1,numel(price_move));
% s(1) = 1;
% for i = 2:numel(price_move)
%     for state = 1:2
%         alpha(state,i) = B(state,obs2ind(price_move(i))) .* (sum(alpha(:,i-1) .*A(:,state)));
%     end
%     s(i) =  sum(alpha(:,i));
%     alpha(:,i) = alpha(:,i)./s(i);
% end

% Backward Algorithm
% Matrix method
beta(:,end) = 1;
for i = numel(price_move)-1:-1:1
    beta(:,i) = A*(B(:,obs2ind(price_move(i+1))).*beta(:,i+1));
end
% Loop Method
% for i = numel(price_move)-1:-1:1
%     for state = 1:2
%       beta(state,i) = (1/s(i+1)) * sum( A(state,:)'.* beta(:,i+1) .* B(:,obs2ind(price_move(i)))); 
%     end
% end

lalpha = log(alpha);
lbeta = log(beta);
% Update parameters

gamma = zeros(2, numel(price_move));
epsilon = zeros(2, 2);

% Expectation
for i = 1:numel(price_move)
    gamma(:,i) = alpha(:,i).*beta(:,i)./sum(alpha(:,end));
%     gamma(:,i) = exp(lalpha(:,i)+lbeta(:,i)- log(sum(alpha(:,end))));
end

for i = 1:numel(price_move)-1
    numerator = A.*(alpha(:,i)*(beta(:,i+1).*B(:,obs2ind(price_move(i+1))))');
    normalization = sum(alpha(:,end));
    epsilon = epsilon + numerator./normalization;
end

% for k = 1:numel(price_move)-1
%     for i = 1:2
%         for j = 1:2
%             epsilon(i,j) = epsilon(i,j) + exp(lalpha(i,k)+lbeta(j,k+1)+log(A(i,j))+log(B(j,obs2ind(price_move(k+1))))-log(sum(alpha(:,end))));
%         end
%     end
% end
% 
% % Maximization
% pi_new = gamma(:,1);
% A_new = bsxfun(@rdivide, epsilon, sum(gamma(:,1:end-1),2));
% y_1 = (price_move == 1)';
% y_1 = repmat(y_1,2,1);
% y_2 = (price_move == -1)';
% y_2 = repmat(y_2,2,1);
% B_new(:,1) = sum(y_1 .* gamma,2)./sum(gamma,2);
% B_new(:,2) = sum(y_2 .* gamma,2)./sum(gamma,2);
% 
% diff = norm(A_new(:) - A(:),1) + norm(B_new(:) - B(:),1);
% iter = iter + 1;
% 
% pi = pi_new;
% A = A_new;
% B = B_new;
% if diff < 0.001
%     converged  = true;
% end
% end
prob = gamma;
plot(prob(1,:),'blue');
hold; plot(prob(2,:),'red');

end

function index = obs2ind(obs)

if(obs == 1)
    index = 1;
else
    index = 2;
end

end
    