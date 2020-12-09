load('mnist_all.mat');
X = train0(1:50, :);
X(51:100,:) = train1(1:50,:);
X(101:150,:) = train2(1:50,:);
X(151:200,:) = train3(1:50,:);

N = 50;
numCols = 10;
numRows = ceil(4*N/numCols);
d = sqrt(size(X,2));

figure;
set(gcf, 'color', 'white');
set(gcf, 'Position', [520 85 1020 720]);
for i=1:size(X,1)
    subplot(numRows,numCols,i);
    img = reshape(X(i,:),d,d)';
    imagesc(img);
    set(gca, 'xtick', []);
    set(gca, 'ytick', []);
end
colormap(gray);

saveas(gcf, 'digit_image.jpg', 'jpeg');

X = double(X);
[U,Z,S] = pca(X);
pc1 = U(:,1);
pc2 = U(:,2);
figure;
set(gcf,'color', 'white');
img1 = reshape(pc1,d,d);
imagesc(img1);
colormap(gray);
saveas(gcf, 'pc1_plot.jpg', 'jpeg');

figure;
set(gcf,'color', 'white');
img2 = reshape(pc2,[d d]);
imagesc(img2);
colormap(gray);
saveas(gcf, 'pc2_plot.jpg', 'jpeg');

figure;
set(gcf, 'color', 'white');
plot(Z(1:50,1),Z(1:50,2),'r*', 'DisplayName', '0'); 
legend
hold on
plot(Z(51:100,1), Z(51:100,2), 'b+', 'DisplayName', '1');
plot(Z(101:150,1), Z(101:150,2), 'ko', 'DisplayName', '2');
plot(Z(151:200,1), Z(151:200,2), 'gv', 'DisplayName', '3');
hold off
saveas(gcf, 'cse881hw1_part_2d.jpg', 'jpeg');
%class 0 and 1 are easily discernable. class 2 and 3 are hard to distinguish

rank = 2;
W = Z(:,1:rank)*diag(S(1:rank))*U(:,1:rank)';

figure;
set(gcf, 'color', 'white');
set(gcf, 'Position', [520 85 1020 720]);
for i=1:size(W,1)
    subplot(numRows,numCols,i);
    img = reshape(W(i,:),d,d)';
    imagesc(img);
    set(gca, 'xtick', []);
    set(gca, 'ytick', []);
end
colormap(gray);

saveas(gcf, 'cse881hw_part_2e', 'jpeg');
%same result

rank = 50;
W = Z(:,1:rank)*diag(S(1:rank))*U(:,1:rank)';

figure;
set(gcf, 'color', 'white');
set(gcf, 'Position', [520 85 1020 720]);
for i=1:size(W,1)
    subplot(numRows,numCols,i);
    img = reshape(W(i,:),d,d)';
    imagesc(img);
    set(gca, 'xtick', []);
    set(gca, 'ytick', []);
end
colormap(gray);

saveas(gcf, 'cse881hw_part_2f', 'jpeg');
%all the digits are easily dicernable with a high rank of 50
