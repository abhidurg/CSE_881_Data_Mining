proximity_matrix = [0, 0.5840, 0.1955, 0.3815, 0.1127; 0.5840, 0, 0.6132, 0.4956, 0.5733; ...
    0.1955 0.6132, 0, 0.2390, 0.3067; 0.3815, 0.4956, 0.2390, 0, 0.4694; 0.1127 0.5733, 0.3067, 0.4694, 0];
v = proximity_matrix(tril(true(size(proximity_matrix)), -1))';
z = linkage(v, 'complete');
dendrogram(z);
xlabel('Points p1 - p5');
ylabel('Distance between clusters');
title('Dendrogram of hierarchical clustering using complete linkage')