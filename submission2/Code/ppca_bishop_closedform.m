%Please see the microsoft paper "Bishop ....."
%Check by variying q size .................

clear;
clc;

%Image file reading 
T = imread('9_1.jpg');
T = rgb2gray(T);
T = im2double(T);
ori = T;

%No of Basis (Eigen Vectors)
q = 100;
 
%Computing the mean value
for j = 1:size(T,2)
    mu(j) = mean(T(:,j));
end

S = zeros(size(T,2));

%Computing the Covariance matrix as mentioned in the first para of the
%paper 
for n = 1:size(T,1)
    S = S + (T(n,:)' - mu') * (T(n,:)' - mu')';
end

S = 1/size(T,1)*S;

%Computing the eigen value and eigen vectors form the covariance matrix
[e_ve,e_v] = eig(S);
e_v = diag(e_v);
[e_v, i] = sort(e_v, 'descend');
e_ve = e_ve(:,i);
U = e_ve(:,1:q);
lambda_diag = diag(e_v);
L = lambda_diag(1:q, 1:q);

%This two parameters have been calculated using equation no 7 and equation
%no 8 from the paper 
sigma = sqrt(1/(size(S,1)-q)*sum(e_v(q+1:size(S,1))));
W = U * sqrt(L - sigma^2*eye(q));


%Value of M calculated equation given on page no 5 above section 3.2 last
%line
M = W'*W + sigma^2 * eye(q);


for i = 1:size(T,1)
     Tnorm(i,:) = T(i,:) - mu;
end

%Equation no 6 
X = inv(M)*W'*Tnorm';

%Mentioned in page no 6 above section 4 last line
rec = ((W*inv(W'*W)*M*X))';

for j = 1:size(T,1)
    rec(j,1:size(T,2)) = (rec(j,1:size(T,2))+mu(1:size(T,2)));
end

%Computing the norm error 
norm(rec-ori)
imshow(rec);
