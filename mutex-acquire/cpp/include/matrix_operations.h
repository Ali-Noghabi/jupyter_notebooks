#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>
#include <string>

typedef std::vector<std::vector<int>> Matrix;

Matrix threaded_matrix_multiply(const Matrix& A, const Matrix& B);
Matrix single_thread_matrix_multiply(const Matrix& A, const Matrix& B);
Matrix load_matrix_from_file(const std::string& filename);

#endif
