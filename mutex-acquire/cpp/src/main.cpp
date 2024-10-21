#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>
#include <filesystem>
#include "matrix_operations.h"
#include "matrices_generator.h"

int main() {
    const int matrix_size_A = 700;
    const int matrix_size_B = 700;
    const std::string file_path_A = "matrices/matrix_A.txt";
    const std::string file_path_B = "matrices/matrix_B.txt";
    const std::string result_file_path = "matrices/result.txt";
    const std::string report_file_path = "matrices/result_report.txt";
    const int num_iterations = 100;

    std::filesystem::create_directories("matrices");

    double total_threaded_time = 0;
    double total_single_thread_time = 0;
    double total_numpy_time = 0;

    std::ofstream report_file(report_file_path);

    for (int i = 0; i < num_iterations; ++i) {
        std::cout << "Iteration " << i + 1 << "/" << num_iterations << std::endl;

        generate_and_save_matrices(matrix_size_A, matrix_size_B, file_path_A, file_path_B);

        auto A = load_matrix_from_file(file_path_A);
        auto B = load_matrix_from_file(file_path_B);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto C_threaded = threaded_matrix_multiply(A, B);
        auto end_time = std::chrono::high_resolution_clock::now();
        double threaded_time = std::chrono::duration<double>(end_time - start_time).count();
        total_threaded_time += threaded_time;

        start_time = std::chrono::high_resolution_clock::now();
        auto C_single_thread = single_thread_matrix_multiply(A, B);
        end_time = std::chrono::high_resolution_clock::now();
        double single_thread_time = std::chrono::duration<double>(end_time - start_time).count();
        total_single_thread_time += single_thread_time;

        assert(C_threaded == C_single_thread);

        report_file << "Iteration " << i + 1 << "\n";
        report_file << "Threaded matrix multiplication time: " << threaded_time << " seconds\n";
        report_file << "Single-threaded matrix multiplication time: " << single_thread_time << " seconds\n\n";
    }

    report_file.close();

    double average_threaded_time = total_threaded_time / num_iterations;
    double average_single_thread_time = total_single_thread_time / num_iterations;

    std::cout << "Average threaded matrix multiplication time: " << average_threaded_time << " seconds" << std::endl;
    std::cout << "Average single-threaded matrix multiplication time: " << average_single_thread_time << " seconds" << std::endl;

    std::ofstream report_file_append(report_file_path, std::ios_base::app);
    report_file_append << "Average times for all iterations\n";
    report_file_append << "Average threaded matrix multiplication time: " << average_threaded_time << " seconds\n";
    report_file_append << "Average single-threaded matrix multiplication time: " << average_single_thread_time << " seconds\n";
    report_file_append.close();

    return 0;
}
