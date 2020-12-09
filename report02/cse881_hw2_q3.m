P = importdata('prices.txt');
rng(1);
[clusters,centroid] = kmeans(P, 10, 'Distance', 'correlation', 'Replicates', 500);
class_labels = importdata('sp500.class');
C = confusionmat(class_labels, clusters);
confusionchart(C);