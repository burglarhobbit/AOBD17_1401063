%Please see the microsoft paper "Bishop ....."
%Check by variying q size .................

clear;
clc;

tic;
%Image file reading 
T = imread('9_1.jpg');
T = rgb2gray(T);
T = im2double(T);
ori = T;
imwrite(ori,'Origianl.jpg');
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


No_iteration = 3;
W = randn(size(T,2),q);
sigma = randn(1);

for i=1:1:No_iteration
    M = W'*W + sigma*eye(q);
    W = S*W*inv(sigma*eye(q)+inv(M)*W'*S*W);
    sigma = (1/size(S,1))*trace(S-S*W*inv(M)*W');
end


for i = 1:size(T,1)
     Tnorm(i,:) = T(i,:) - mu;
end

%Equation no 6 
X = W'*Tnorm';

%Mentioned in page no 6 above section 4 last line
rec = ((W*inv(W'*W)*X))';

for j = 1:size(T,1)
    rec(j,1:size(T,2)) = (rec(j,1:size(T,2))+mu(1:size(T,2)));
end

%Computing the norm error 
error = norm(rec-ori);
imshow(rec);
imwrite(rec,'New_Image.jpg');
toc;