#include "matrices_generator.h"
#include <random>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include "matrix_operations.h"

void generate_and_save_matrices(int matrix_size_A, int matrix_size_B, const std::string& file_path_A, const std::string& file_path_B) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 9);

    std::ofstream file_A(file_path_A);
    std::ofstream file_B(file_path_B);

    for (int i = 0; i < matrix_size_A; ++i) {
        for (int j = 0; j < matrix_size_B; ++j) {
            file_A << dist(rng) << " ";
        }
        file_A << "\n";
    }

    for (int i = 0; i < matrix_size_B; ++i) {
        for (int j = 0; j < matrix_size_A; ++j) {
            file_B << dist(rng) << " ";
        }
        file_B << "\n";
    }
}

double multiply_matrices(const std::string& file_path_A, const std::string& file_path_B, const std::string& result_file_path) {
    auto A = load_matrix_from_file(file_path_A);
    auto B = load_matrix_from_file(file_path_B);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = single_thread_matrix_multiply(A, B);
    auto end_time = std::chrono::high_resolution_clock::now();
    double multiplication_time = std::chrono::duration<double>(end_time - start_time).count();

    std::ofstream result_file(result_file_path);
    for (const auto& row : result) {
        for (const auto& val : row) {
            result_file << val << " ";
        }
        result_file << "\n";
    }

    std::ofstream report_file(result_file_path + "_report.txt", std::ios_base::app);
    report_file << "Matrix multiplication using numpy took: " << multiplication_time << " seconds\n";

    return multiplication_time;
}
