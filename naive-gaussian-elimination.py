import numpy as np

def gaussian_elimination(matrix, b):
    n = len(b)
    
    # Augment the matrix with the vector b
    A = np.hstack([matrix, b.reshape(-1, 1)])

    # Forward elimination
    for i in range(n):
        # Find the maximum element in the current column for pivoting
        max_row = np.argmax(np.abs(A[i:n, i])) + i
        
        # Check if the pivot element is zero
        if A[max_row, i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")
        
        # Swap the current row with the max_row if needed
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
        
        # Normalize the pivot row
        A[i] = A[i] / A[i][i]
        
        # Eliminate the entries below the pivot
        for j in range(i + 1, n):
            A[j] = A[j] - A[i] * A[j][i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] - np.dot(A[i][i+1:n], x[i+1:n])
    
    return x

#2x2
matrix_2x2 = np.array([[2, 1], [5, 7]])
b_2x2 = np.array([11, 13])
solution_2x2 = gaussian_elimination(matrix_2x2, b_2x2)
print("Solution for 2x2 matrix:", solution_2x2) 

#3x3
matrix_3x3 = np.array([[1, 2, -1], [2, 3, 3], [3, -1, 2]])
b_3x3 = np.array([8, 13, 1])
solution_3x3 = gaussian_elimination(matrix_3x3, b_3x3)
print("Solution for 3x3 matrix:", solution_3x3)

#4x4
matrix_4x4 = np.array([[1, 1, 1, 1], [0, 2, 5, 1], [2, 5, -1, 0], [1, -1, 1, 1]])
b_4x4 = np.array([10, 12, 6, 4])
solution_4x4 = gaussian_elimination(matrix_4x4, b_4x4)
print("Solution for 4x4 matrix:", solution_4x4)