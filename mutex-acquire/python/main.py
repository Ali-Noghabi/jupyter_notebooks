import ctypes
from ctypes import c_int
import threading
import numpy as np
import time
import os
from matrices_generator import generate_and_save_matrices, multiply_matrices

# Define a compare and swap function using ctypes
def compare_and_swap(address, old, new):
    if address.contents.value == old:
        address.contents.value = new
        return True
    else:
        return False

class Mutex:
    def __init__(self):
        # Mutex state, 0 means unlocked, 1 means locked
        self.lock = ctypes.pointer(c_int(0))

    def acquire(self):
        while True:
            # Try to set the lock from 0 (unlocked) to 1 (locked)
            if compare_and_swap(self.lock, 0, 1):
                return

    def release(self):
        # Set the lock to 0 (unlocked)
        self.lock.contents.value = 0

def matrix_multiply(A, B, C, row, mutex):
    for j in range(B.shape[1]):
        sum = 0
        for k in range(A.shape[1]):
            sum += A[row, k] * B[k, j]
        mutex.acquire()
        C[row, j] = sum
        mutex.release()

def threaded_matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible matrices for multiplication"

    C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
    mutex = Mutex()

    threads = []
    for i in range(A.shape[0]):
        t = threading.Thread(target=matrix_multiply, args=(A, B, C, i, mutex))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return C

def single_thread_matrix_multiply(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible matrices for multiplication"
    C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            sum = 0
            for k in range(A.shape[1]):
                sum += A[i, k] * B[k, j]
            C[i, j] = sum
    return C

def load_matrix_from_file(filename):
    return np.loadtxt(filename, dtype=int)

# Parameters
matrix_size_A = 300  # Number of rows in matrix A
matrix_size_B = 400  # Number of columns in matrix A and rows in matrix B
file_path_A = "matrices/matrix_A.txt"
file_path_B = "matrices/matrix_B.txt"
result_file_path = "matrices/result.txt"
report_file_path = "matrices/result_report.txt"
num_iterations = 5

# Ensure the directory exists
os.makedirs(os.path.dirname(report_file_path), exist_ok=True)

# Timing accumulators
total_threaded_time = 0
total_single_thread_time = 0
total_numpy_time = 0

with open(report_file_path, 'w') as report_file:
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")

        # Generate and save matrices
        generate_and_save_matrices(matrix_size_A, matrix_size_B, file_path_A, file_path_B)

        # Load matrices from file
        A = load_matrix_from_file(file_path_A)
        B = load_matrix_from_file(file_path_B)

        # Perform numpy-based matrix multiplication and save result to file (for validation)
        numpy_time = multiply_matrices(file_path_A, file_path_B, result_file_path)
        total_numpy_time += numpy_time

        # Load result from result.txt
        result_from_file = load_matrix_from_file(result_file_path)

        # Perform matrix multiplication with threading
        start_time = time.time()
        C_threaded = threaded_matrix_multiply(A, B)
        end_time = time.time()
        threaded_time = end_time - start_time
        total_threaded_time += threaded_time

        # Perform matrix multiplication without threading
        start_time = time.time()
        C_single_thread = single_thread_matrix_multiply(A, B)
        end_time = time.time()
        single_thread_time = end_time - start_time
        total_single_thread_time += single_thread_time

        # Compare results
        assert np.array_equal(C_threaded, C_single_thread), "The results are different!"
        assert np.array_equal(C_threaded, result_from_file), "The result from the file is different!"
        
        print("Results are the same for all methods.")

        # Log the times for this iteration
        report_file.write(f"Iteration {i+1}\n")
        report_file.write(f"Time using numpy: {numpy_time:.4f} seconds\n")
        report_file.write(f"Threaded matrix multiplication time: {threaded_time:.4f} seconds\n")
        report_file.write(f"Single-threaded matrix multiplication time: {single_thread_time:.4f} seconds\n")
        report_file.write("\n")

# Calculate average times
average_threaded_time = total_threaded_time / num_iterations
average_single_thread_time = total_single_thread_time / num_iterations
average_numpy_time = total_numpy_time / num_iterations

# Output average times
print(f"Average numpy matrix multiplication time: {average_numpy_time:.4f} seconds")
print(f"Average threaded matrix multiplication time: {average_threaded_time:.4f} seconds")
print(f"Average single-threaded matrix multiplication time: {average_single_thread_time:.4f} seconds")

# Append the average timing results to result_report.txt
with open(report_file_path, 'a') as report_file:
    report_file.write("Average times for all iterations\n")
    report_file.write(f"Average numpy matrix multiplication time: {average_numpy_time:.4f} seconds\n")
    report_file.write(f"Average threaded matrix multiplication time: {average_threaded_time:.4f} seconds\n")
    report_file.write(f"Average single-threaded matrix multiplication time: {average_single_thread_time:.4f} seconds\n")
