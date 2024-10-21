import os
import numpy as np
import time

def generate_and_save_matrices(matrix_size_A, matrix_size_B, file_path_A, file_path_B):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path_A), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_B), exist_ok=True)

    # Generate random matrices
    matrix_A = np.random.randint(10, size=(matrix_size_A, matrix_size_B))
    matrix_B = np.random.randint(10, size=(matrix_size_B, matrix_size_A))

    # Save matrices to file
    with open(file_path_A, 'w') as file_A:
        for row in matrix_A:
            file_A.write(' '.join(map(str, row)) + '\n')

    with open(file_path_B, 'w') as file_B:
        for row in matrix_B:
            file_B.write(' '.join(map(str, row)) + '\n')


def multiply_matrices(file_path_A, file_path_B, result_file_path):
    # Load matrices from file
    with open(file_path_A, 'r') as file_A:
        matrix_A = [list(map(int, line.split())) for line in file_A]

    with open(file_path_B, 'r') as file_B:
        matrix_B = [list(map(int, line.split())) for line in file_B]

    if len(matrix_A[0]) != len(matrix_B):
        print("Matrices are not multiplicatively compatible.")
        return

    # Perform matrix multiplication with numpy
    start_time = time.time()
    result = np.dot(matrix_A, matrix_B)
    end_time = time.time()
    multiplication_time = end_time - start_time

    # Save result matrix to file
    with open(result_file_path, 'w') as result_file:
        for row in result:
            result_file.write(' '.join(map(str, row)) + '\n')

    # Append multiplication time to report file
    report_file_path = result_file_path.split('.')[0] + "_report.txt"
    with open(report_file_path, 'a') as report_file:
        report_file.write("Matrix multiplication using numpy took: {} seconds\n".format(multiplication_time))

    return multiplication_time

# Example usage in main.py:
# matrix_size_A = 300  # Number of rows in matrix A
# matrix_size_B = 400  # Number of columns in matrix A and rows in matrix B
# file_path_A = "matrices/matrix_A.txt"
# file_path_B = "matrices/matrix_B.txt"
# result_file_path = "matrices/result.txt"
#
# generate_and_save_matrices(
#     matrix_size_A, matrix_size_B, file_path_A, file_path_B)
#
# multiply_matrices(file_path_A, file_path_B, result_file_path)
