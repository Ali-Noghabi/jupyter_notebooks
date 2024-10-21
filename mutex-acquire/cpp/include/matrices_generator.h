#ifndef MATRICES_GENERATOR_H
#define MATRICES_GENERATOR_H

#include <string>

void generate_and_save_matrices(int matrix_size_A, int matrix_size_B, const std::string& file_path_A, const std::string& file_path_B);
double multiply_matrices(const std::string& file_path_A, const std::string& file_path_B, const std::string& result_file_path);

#endif
