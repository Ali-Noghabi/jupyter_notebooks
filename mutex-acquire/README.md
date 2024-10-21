
# Matrix Multiplication Performance Analysis

## Project Overview
This project aims to analyze the performance of matrix multiplication using three different methods: numpy, single-threaded, and multi-threaded with mutex locks in Python. Additionally, the project includes a C++ implementation for comparison. The primary goal is to compare the efficiency and correctness of these methods.

## Table of Contents

  - [What is Mutex and Acquire Function](#what-is-mutex-and-acquire-function)
  - [Mutex Implementation](#mutex-implementation)
  - [Single-threaded Matrix Multiplication](#single-threaded-matrix-multiplication)
  - [Multi-threaded Matrix Multiplication](#multi-threaded-matrix-multiplication)
  - [Analysis of Reports](#analysis-of-reports)
  - [Performance Comparison](#performance-comparison)
  - [Conclusion](#conclusion)

## What is Mutex and Acquire Function
A **mutex** (mutual exclusion) is a synchronization primitive used to prevent concurrent threads from accessing shared resources simultaneously. It ensures that only one thread can access the resource at any given time, thereby avoiding race conditions.

![mutex](mutex.webp)

## Mutex Implementation

### python
the mutex is implemented using the `ctypes` library to create a compare-and-swap function. The `Mutex` class has `acquire` and `release` methods to manage the lock state. Threads use the `acquire` method to lock the mutex before accessing shared resources and `release` to unlock it afterward.

```python
class Mutex:
    def __init__(self):
        self.lock = ctypes.pointer(c_int(0))

    def acquire(self):
        while True:
            if compare_and_swap(self.lock, 0, 1):
                return

    def release(self):
        self.lock.contents.value = 0
```

implements a mutex using atomic operations. The `acquire` method repeatedly attempts to set the `lock` to `true` if it is currently `false`, effectively locking the mutex. The `release` method sets the `lock` to `false`, unlocking the mutex for other threads to use. This ensures that only one thread can access the critical section of code at a time, providing thread safety.
```cpp
Mutex::Mutex() : lock(false) {}

void Mutex::acquire() {
    bool expected = false;
    while (!lock.compare_exchange_strong(expected, true)) {
        expected = false;  // Reset expected value for the next iteration
    }
}

void Mutex::release() {
    lock.store(false);
}
```
## Single-threaded Matrix Multiplication
The single-threaded matrix multiplication method involves straightforward nested loops to perform the matrix multiplication. Each element of the resulting matrix is computed one by one without any parallelization.

![matrix multiplication](matrix%20multiplication.webp)
### python
```python
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
```

### cpp
```cpp
Matrix single_thread_matrix_multiply(const Matrix& A, const Matrix& B) {
    assert(A[0].size() == B.size());

    Matrix C(A.size(), std::vector<int>(B[0].size(), 0));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B[0].size(); ++j) {
            int sum = 0;
            for (size_t k = 0; k < A[0].size(); ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}
```

## Multi-threaded Matrix Multiplication

### python
The `threaded_matrix_multiply` function performs matrix multiplication using multiple threads. It initializes a result matrix `C` and a mutex for thread safety, then creates and starts a thread for each row of the result matrix. Each thread computes one row of the result matrix, and the main thread waits for all threads to finish. Finally, it returns the computed result matrix `C`.
```python
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
```

### cpp
The `threaded_matrix_multiply` function performs matrix multiplication using multiple threads in C++. It initializes a result matrix `C` and creates a thread for each row of matrix `A`, each computing the corresponding row in the result matrix. The main thread waits for all worker threads to complete their execution using `join`. Finally, the function returns the computed result matrix `C`.

```cpp
Matrix threaded_matrix_multiply(const Matrix& A, const Matrix& B) {
    assert(A[0].size() == B.size());

    Matrix C(A.size(), std::vector<int>(B[0].size(), 0));
    std::vector<std::thread> threads;
    threads.reserve(A.size());

    for (size_t i = 0; i < A.size(); ++i) {
        threads.emplace_back(matrix_multiply, std::cref(A), std::cref(B), std::ref(C), i);
    }

    for (auto& t : threads) {
        t.join();
    }

    return C;
}
```
## Analysis of Reports

### python
The following data summarizes the time taken by each method over 100 iterations for:
#### $A_{300 \times 400}$ $\times$ $B_{400 \times 300}$

- **Average numpy matrix multiplication time:** 0.0253 seconds
- **Average threaded matrix multiplication time:** 7.9636 seconds
- **Average single-threaded matrix multiplication time:** 7.6882 seconds

#### Iteration Reports
Here is an example of the detailed report for each iteration:

```
Iteration 1
Time using numpy: 0.0285 seconds
Threaded matrix multiplication time: 7.5576 seconds
Single-threaded matrix multiplication time: 8.9681 seconds
...
Iteration 100
Time using numpy: 0.0120 seconds
Threaded matrix multiplication time: 7.8053 seconds
Single-threaded matrix multiplication time: 7.6021 seconds
```

### cpp
The following data summarizes the time taken by each method over 100 iterations for:
#### $A_{700 \times 700}$ $\times$ $B_{700 \times 700}$

- **Average threaded matrix multiplication time:** 0.0748568 seconds
- **Average single-threaded matrix multiplication time:** 0.264542 seconds seconds

## Performance Comparison

### Python vs. C++
- **Python (numpy):**
  - **Average time:** 0.0253 seconds
  - **Advantages:** Optimized libraries, hardware acceleration, memory efficiency, and algorithm optimization.
  - **Disadvantages:** Limited by GIL in multi-threaded scenarios.

- **C++ (multi-threaded):**
  - **Average time:** 0.0748568 seconds
  - **Advantages:** True parallelism, lower overhead for thread management, and efficient use of system resources.
  - **Disadvantages:** More complex implementation and potential for race conditions without proper synchronization.

### Why numpy Works Very Well
Numpy performs very well for matrix multiplication due to several reasons:
1. **Optimized Libraries:** Numpy leverages highly optimized libraries like BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra Package), which are written in low-level languages like C and Fortran. These libraries are specifically designed for efficient numerical computations.
2. **Hardware Acceleration:** Numpy can utilize hardware acceleration features available on modern CPUs, such as vectorized operations and parallel processing, to speed up computations.
3. **Memory Efficiency:** Numpy uses contiguous memory blocks and optimized memory access patterns, reducing cache misses and improving data access speeds.
4. **Algorithm Optimization:** Numpy implements highly efficient algorithms for matrix multiplication, taking advantage of mathematical properties and advanced techniques to minimize computation time.

### Overhead of Thread Management
The threaded implementation does not show a significant improvement over the single-threaded method due to several reasons:
1. **Thread Creation and Management:** Creating and managing multiple threads incurs overhead. Each thread requires its own stack space and resources, leading to increased memory usage and context-switching overhead.
2. **Mutex Locking:** The use of mutex locks to ensure thread safety adds synchronization overhead. Threads spend time waiting for locks to be acquired and released, which can negate the benefits of parallelism.
3. **GIL (Global Interpreter Lock):** In CPython, the Global Interpreter Lock (GIL) ensures that only one thread executes Python bytecode at a time, limiting the potential for true parallelism in multi-threaded Python programs.
4. **Granularity of Tasks:** Matrix multiplication involves fine-grained tasks where each element calculation is relatively quick. The overhead of managing threads and locks can overshadow the benefits of parallel execution for such fine-grained tasks.

### Why Numpy is Better Than Both Threaded and Single-threaded Methods
1. **Efficiency of Optimized Libraries:** Numpy's use of optimized libraries and hardware acceleration results in much faster computations compared to manually implemented methods.
2. **Reduced Overhead:** Numpy's matrix multiplication avoids the overhead associated with thread creation, context switching, and mutex locking, leading to more efficient execution.
3. **Parallelism and Vectorization:** Numpy can take advantage of low-level parallelism and vectorized operations, providing significant speedup compared to high-level threading in Python.

## Conclusion
The C++ implementation, especially in a multi-threaded environment, shows significant performance improvements over the single-threaded version and highlights the power of parallel computing in C++. While numpy in Python is highly optimized for matrix operations and performs exceptionally well, C++ offers better control over multi-threading and can achieve superior performance for computationally intensive tasks.
