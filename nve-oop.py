import numpy as np

class GaussianElimination:
    def __init__(self, matrix, b):
        self.matrix = matrix
        self.b = b
        self.n = len(b)
        self.augmented_matrix = np.hstack([self.matrix, self.b.reshape(-1, 1)])

    def forward_elimination(self):
        for i in range(self.n):
            self.augmented_matrix[i] = self.augmented_matrix[i] / self.augmented_matrix[i][i]

            for j in range(i + 1, self.n):
                self.augmented_matrix[j] -= self.augmented_matrix[i] * self.augmented_matrix[j][i]

    def back_substitution(self):
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = self.augmented_matrix[i][self.n] - np.dot(self.augmented_matrix[i][i+1:self.n], x[i+1:self.n])
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