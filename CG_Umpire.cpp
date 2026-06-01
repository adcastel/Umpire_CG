#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <stdexcept>


#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/NumaPolicy.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/numa.hpp"


umpire::ResourceManager* rm{nullptr};
umpire::Allocator host_alloc;
// static std::unordered_map<int, umpire::Allocator> numa_alloc_cache;
inline static std::vector<umpire::Allocator> alloc_cache;

enum class MemType { UNKNOWN, DDR, HBM };
std::unordered_map<int, int> node_type;
int pool_type = 0;  // 1 DDR, 2 HBM

void initialize_allocators(){

    auto allocatable_nodes = umpire::numa::get_allocatable_nodes();
    alloc_cache.resize(allocatable_nodes.size());
    for (int i = 0; i < allocatable_nodes.size(); i++) {
        auto alloc = rm->makeAllocator<umpire::strategy::NumaPolicy>(
            "host_alloc_" + std::to_string(i), rm->getAllocator("HOST"),
            allocatable_nodes[i]);

        alloc_cache[i] = alloc;
    }

}

bool node_has_no_cpus(int node)
{
    std::string path =
        "/sys/devices/system/node/node" + std::to_string(node) + "/cpulist";

    std::ifstream file(path);

    if (!file.is_open()) {
        return false;
    }

    std::string cpulist;
    std::getline(file, cpulist);

    // Remove whitespace
    cpulist.erase(std::remove_if(cpulist.begin(), cpulist.end(), ::isspace),
                  cpulist.end());
    return cpulist.empty();
}


void set_memory_type(){
     auto allocatable_nodes = umpire::numa::get_allocatable_nodes();
    for (int node = 0; node < allocatable_nodes.size(); node++) {
        std::cout << "Node " << node << " is ";

        if (node_has_no_cpus(node)) {
            node_type[node] = 2;  // HBM
            std::cout << "HBM" << std::endl;
        } else {
            std::cout << "DDR" << std::endl;
            node_type[node] = 1;  // DDR
        }
    }
}
int select_pool(){
    const char* val = std::getenv("MEM_ALLOC");

    if (val != nullptr) {
        return std::atoi(val);
    } else {
        return 0;
    }
}
int select_node(){
    
    const char* val = std::getenv("NODE_ALLOC");

    if (val != nullptr) {
        return std::atoi(val);
    } else {
        return 0;
    }

}

void * allocate_memory(int node, size_t num_bytes, std::string part) {
    std::cout<<"Allocating "<< part <<" with memory size of "<< num_bytes <<" bytes on node "<< node << " that is "<< (node_type[node] == 1 ? "DDR" : "HBM") << std::endl;
    host_alloc = alloc_cache[node];
    return host_alloc.allocate(num_bytes);
}

class CSRMatrix {
public:
    int n;
    int nnz;

    //std::vector<int> row_ptr;
    int * row_ptr;
    //std::vector<int> col_idx;
    int * col_idx;
    //std::vector<double> values;
    double * values;

    CSRMatrix(const std::string& filename, int nodo=0)
    {
        std::ifstream file(filename);

        if (!file)
            throw std::runtime_error("Cannot open matrix file");

        file >> n >> nnz;

        //row_ptr.resize(n + 1);
        int node_row = std::atoi(std::getenv("ROW_PTR"));
        row_ptr = static_cast<int*>(allocate_memory(node_row, (n + 1) * sizeof(int), "ROW_PTR"));
        //col_idx.resize(nnz);
        int node_col = std::atoi(std::getenv("COL_IDX"));
        col_idx = static_cast<int*>(allocate_memory(node_col, nnz * sizeof(int), "COL_IDX"));
        //values.resize(nnz);
        int node_val = std::atoi(std::getenv("VALUES"));
        values = static_cast<double*>(allocate_memory(node_val, nnz * sizeof(double), "VALUES"));

        for (int i = 0; i < n + 1; i++)
            file >> row_ptr[i];

        for (int i = 0; i < nnz; i++)
            file >> col_idx[i];

        for (int i = 0; i < nnz; i++)
            file >> values[i];
    }

    void matvec(
        const double * x,
        double * y) const
    {
        for (int i = 0; i < n; i++) {
            y[i] = 0.0;
        }

        for (int i = 0; i < n; i++) {

            for (int j = row_ptr[i];
                 j < row_ptr[i + 1];
                 j++) {

                y[i] += values[j] *
                        x[col_idx[j]];
            }
        }
    }
};

double dot(
    const double * a,
    const double * b, int an)
{
    double sum = 0.0;

    for (size_t i = 0; i < an; i++)
        sum += a[i] * b[i];

    return sum;
}

double norm2(const double * x, int n)
{
    return std::sqrt(dot(x, x, n));
}

/***************************************************
 * Jacobi preconditioner
 ***************************************************/
class JacobiPreconditioner {
public:
    std::vector<double> inv_diag;

    explicit JacobiPreconditioner(
        const CSRMatrix& A)
    {
        inv_diag.resize(A.n);

        for (int i = 0; i < A.n; i++) {

            double diag = 0.0;

            for (int j = A.row_ptr[i];
                 j < A.row_ptr[i + 1];
                 j++) {

                if (A.col_idx[j] == i) {
                    diag = A.values[j];
                    break;
                }
            }

            inv_diag[i] = 1.0 / diag;
        }
    }

    void apply(
        const double * r,
        double * z, int An) const
    {
        for (size_t i = 0; i < An; i++)
            z[i] = inv_diag[i] * r[i];
    }
};

/***************************************************
 * Preconditioned Conjugate Gradient
 ***************************************************/
int pcg(
    const CSRMatrix& A,
    const double * b,
    double * x,
    int max_iter,
    double tol)
{
    const int n = A.n;

    JacobiPreconditioner M(A);

    int node_r = std::atoi(std::getenv("R"));
    double * r = static_cast<double*>(allocate_memory(node_r, n * sizeof(double), "R"));

    int node_z = std::atoi(std::getenv("Z"));
    double * z = static_cast<double*>(allocate_memory(node_z, n * sizeof(double), "Z"));

    int node_p = std::atoi(std::getenv("P"));
    double * p = static_cast<double*>(allocate_memory(node_p, n * sizeof(double), "P"));

    int node_Ap = std::atoi(std::getenv("AP"));
    double * Ap = static_cast<double*>(allocate_memory(node_Ap, n * sizeof(double), "AP"));
    // r = b - A*x
    A.matvec(x, Ap);

    for (int i = 0; i < n; i++)
        r[i] = b[i] - Ap[i];

    // z = M^-1 r
    M.apply(r, z, n);

    //p = z;
    for (int i = 0; i < n; i++)
        p[i] = z[i];

    double rz_old = dot(r, z, n);

    for (int iter = 0; iter < max_iter; iter++) {

        A.matvec(p, Ap);

        double alpha =
            rz_old / dot(p, Ap, n);

        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];

        for (int i = 0; i < n; i++)
            r[i] -= alpha * Ap[i];

        double residual = norm2(r,n);

        std::cout
            << "Iter "
            << std::setw(4)
            << iter
            << " residual = "
            << std::scientific
            << residual
            << "\n";

        if (residual < tol)
            return iter + 1;

        M.apply(r, z, n);

        double rz_new = dot(r, z, n);

        double beta =
            rz_new / rz_old;

        for (int i = 0; i < n; i++)
            p[i] = z[i] + beta * p[i];

        rz_old = rz_new;
    }

    return max_iter;
}

double compute_residual(
    const CSRMatrix& A,
    const double * x,
    const double * b)
{
    int node_ax = std::atoi(std::getenv("AX"));
    double * Ax = static_cast<double*>(allocate_memory(node_ax, A.n * sizeof(double), "AX"));

    A.matvec(x, Ax);

    for (size_t i = 0; i < A.n; i++)
        Ax[i] -= b[i];

    return norm2(Ax, A.n);
}

int main(int argc, char *argv[])
{
    try {
        
        //pool_type = select_pool();
        rm = &umpire::ResourceManager::getInstance();
        auto allocatable_nodes = umpire::numa::get_allocatable_nodes();

        if (alloc_cache.size() == 0) {
            std::cout << "Allocatable Nodes with or without assigned CPUs: "
                      << allocatable_nodes.size() << std::endl;
            initialize_allocators();
            set_memory_type();
        }

        int nodo_matrix = std::atoi(std::getenv("VALUES"));
        auto  matrix = argc > 1 ? argv[1] : "matrix.csr";
        std::cout << "Reading matrix from file: " << matrix << std::endl;
        CSRMatrix A(matrix, nodo_matrix);

        //std::vector<double> b(A.n, 1.0);
        int nodo_b = std::atoi(std::getenv("B"));       
        double * b = static_cast<double*>(allocate_memory(nodo_b, A.n * sizeof(double), "B"));
        for (int i = 0; i < A.n; i++) {
            b[i] = 1.0;
        }
   
        int nodo_x = std::atoi(std::getenv("X"));
        double * x = static_cast<double*>(allocate_memory(nodo_x, A.n * sizeof(double), "X"));
        for (int i = 0; i < A.n; i++) {
            x[i] = 0.0;
        }

        int iterations = pcg(
            A,
            b,
            x,
            1000,
            1e-10);

        std::cout
            << "\nPCG converged in "
            << iterations
            << " iterations\n";

        std::cout << "\nSolution:\n";

        for (int i = 0; i < A.n; i++) {

            std::cout
                << "x[" << i << "] = "
                << std::setprecision(12)
                << x[i]
                << "\n";
        }

        double error =
            compute_residual(A, x, b);

        std::cout
            << "\nFinal residual ||Ax-b|| = "
            << std::scientific
            << error
            << "\n";
    }
    catch (const std::exception& e) {

        std::cerr
            << "Error: "
            << e.what()
            << "\n";

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}