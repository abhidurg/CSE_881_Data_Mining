G = graph([1 1 2 3 4 6 6 5], [2 3 3 4 6 5 7 7]);
%plot(G);
A = adjacency(G);
F = full(A);
deg = degree(G);
D = diag(deg);
Laplacian_matrix = D - F;
[eigenvectors, eigenvalues] = eig(Laplacian_matrix);

[clusters, centroid] = kmeans(eigenvectors(:,1:3), 3);