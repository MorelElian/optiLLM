#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iomanip>

float l2_norm(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

float max_relative_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_rel = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i]) > 1e-12f) {
            float rel = std::abs(a[i] - b[i]) / std::abs(a[i]);
            max_rel = std::max(max_rel, rel);
        }
    }
    return max_rel;
}

std::vector<float> read_vector(const std::string& filename, size_t count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        std::exit(1);
    }

    std::vector<float> vec(count);
    file.read(reinterpret_cast<char*>(vec.data()), count * sizeof(float));
    if (!file) {
        std::cerr << "Error: Failed to read expected number of floats from " << filename << std::endl;
        std::exit(1);
    }

    return vec;
}

void print_all_entries(const std::vector<float>& a, const std::vector<float>& b, float threshold = 0.0f) {
    std::cout << "\nDetailed element-wise comparison:\n";
    std::cout << std::setw(8) << "Index"
              << std::setw(16) << "A[i]"
              << std::setw(16) << "B[i]"
              << std::setw(16) << "Abs diff"
              << std::setw(16) << "Rel diff"
              << std::setw(12) << "Changed"
              << "\n";

    for (size_t i = 0; i < 10; ++i) {
        float abs_diff = std::abs(a[i] - b[i]);
        float rel_diff = std::abs(a[i]) > 1e-12f ? abs_diff / std::abs(a[i]) : std::numeric_limits<float>::infinity();
        bool changed = abs_diff > threshold;

        std::cout << std::setw(8) << i
                  << std::setw(16) << a[i]
                  << std::setw(16) << b[i]
                  << std::setw(16) << abs_diff
                  << std::setw(16) << rel_diff
                  << std::setw(12) << (changed ? "yes" : "no")
                  << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <size> <file1> <file2>" << std::endl;
        return 1;
    }

    size_t size = std::stoul(argv[1]);
    std::string file1 = argv[2];
    std::string file2 = argv[3];

    auto vec1 = read_vector(file1, size);
    auto vec2 = read_vector(file2, size);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "L2 norm       : " << l2_norm(vec1, vec2) << std::endl;
    std::cout << "Max abs diff  : " << max_abs_diff(vec1, vec2) << std::endl;
    std::cout << "Max rel diff  : " << max_relative_diff(vec1, vec2) << std::endl;

    print_all_entries(vec1, vec2);

    return 0;
}
