import numpy as np

class GaussianElimination:
    def __init__(self, matrix, b):
        self.matrix = matrix
        self.b = b
        self.n = len(b)
        self.augmented_matrix = np.hstack([self.matrix, self.b.reshape(-1, 1)])

    def forward_elimination(self):
        for i in range(self.n):
            # Find the maximum element in the current column for pivoting
            max_row = np.argmax(np.abs(self.augmented_matrix[i:self.n, i])) + i
            
            # Check if the pivot element is zero
            if self.augmented_matrix[max_row, i] == 0:
                raise ValueError("Matrix is singular and cannot be solved.")
            
            # Swap the current row with the max_row if needed
            if max_row != i:
                self.augmented_matrix[[i, max_row]] = self.augmented_matrix[[max_row, i]]
            
            # Normalize the pivot row
            self.augmented_matrix[i] = self.augmented_matrix[i] / self.augmented_matrix[i][i]
            
            # Eliminate the entries below the pivot
            for j in range(i + 1, self.n):
                self.augmented_matrix[j] -= self.augmented_matrix[i] * self.augmented_matrix[j][i]

    def back_substitution(self):
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = self.augmented_matrix[i][self.n] - np.dot(self.augmented_matrix[i][i + 1:self.n], x[i + 1:self.n])
        return x

    def solve(self):
        self.forward_elimination()
        return self.back_substitution()

#2x2 Matrix
matrix_2x2 = np.array([[2, 1], [5, 7]])
b_2x2 = np.array([11, 13])
solver_2x2 = GaussianElimination(matrix_2x2, b_2x2)
solution_2x2 = solver_2x2.solve()
print("Solution for 2x2 matrix:", solution_2x2)

#3x3 Matrix
matrix_3x3 = np.array([[1, 2, -1], [2, 3, 3], [3, -1, 2]])
b_3x3 = np.array([8, 13, 1])
solver_3x3 = GaussianElimination(matrix_3x3, b_3x3)
solution_3x3 = solver_3x3.solve()
print("Solution for 3x3 matrix:", solution_3x3)

#4x4 Matrix
matrix_4x4 = np.array([[1, 1, 1, 1], [0, 2, 5, 1], [2, 5, -1, 0], [1, -1, 1, 1]])
b_4x4 = np.array([10, 12, 6, 4])
solver_4x4 = GaussianElimination(matrix_4x4, b_4x4)
solution_4x4 = solver_4x4.solve()
print("Solution for 4x4 matrix:", solution_4x4)