#include "matrix_operations.h"
#include "Mutex.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <cassert>
#include <atomic>

void matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C, int row) {
    for (size_t j = 0; j < B[0].size(); ++j) {
        int sum = 0;
        for (size_t k = 0; k < A[0].size(); ++k) {
            sum += A[row][k] * B[k][j];
        }
        C[row][j] = sum;
    }
}

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

Matrix load_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    Matrix matrix;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int value;
        std::vector<int> row;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }
    return matrix;
}
