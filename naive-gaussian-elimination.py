import numpy as np

def gaussian_elimination(matrix, b):
    n = len(b)
    
    # Augment the matrix with the vector b
    A = np.hstack([matrix, b.reshape(-1, 1)])
    
    # Forward elimination
    for i in range(n):
        # Make the diagonal contain all 1s
        A[i] = A[i] / A[i][i]
        
        for j in range(i + 1, n):
            A[j] = A[j] - A[i] * A[j][i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] - np.dot(A[i][i+1:n], x[i+1:n])
    
    return x