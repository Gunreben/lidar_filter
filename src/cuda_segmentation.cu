#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <math.h>

__device__ float calculateDistance(float* point, float* plane_coefficients)
{
    return fabsf(plane_coefficients[0] * point[0] + 
                 plane_coefficients[1] * point[1] + 
                 plane_coefficients[2] * point[2] + 
                 plane_coefficients[3]) / 
           sqrtf(plane_coefficients[0] * plane_coefficients[0] + 
                 plane_coefficients[1] * plane_coefficients[1] + 
                 plane_coefficients[2] * plane_coefficients[2]);
}

__global__ void findInliersKernel(float* points, int num_points, 
                                 float* plane_coefficients, int* indices,
                                 float distance_threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float* point = &points[idx * 4];
    float distance = calculateDistance(point, plane_coefficients);
    indices[idx] = (distance <= distance_threshold) ? 1 : 0;
}

__global__ void fitPlaneKernel(float* points, int* indices, int num_points,
                              float* plane_coefficients)
{
    // Using basic least squares method for plane fitting
    __shared__ float sums[6]; // xx, xy, xz, yy, yz, zz
    __shared__ float centroid[3];
    
    if (threadIdx.x < 6) sums[threadIdx.x] = 0.0f;
    if (threadIdx.x < 3) centroid[threadIdx.x] = 0.0f;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points && indices[idx] == 1) {
        float* point = &points[idx * 4];
        atomicAdd(&centroid[0], point[0]);
        atomicAdd(&centroid[1], point[1]);
        atomicAdd(&centroid[2], point[2]);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int inlier_count = 0;
        for (int i = 0; i < num_points; i++) {
            if (indices[i] == 1) inlier_count++;
        }
        if (inlier_count > 0) {
            centroid[0] /= inlier_count;
            centroid[1] /= inlier_count;
            centroid[2] /= inlier_count;
        }
    }
    __syncthreads();

    if (idx < num_points && indices[idx] == 1) {
        float* point = &points[idx * 4];
        float dx = point[0] - centroid[0];
        float dy = point[1] - centroid[1];
        float dz = point[2] - centroid[2];
        
        atomicAdd(&sums[0], dx * dx);  // xx
        atomicAdd(&sums[1], dx * dy);  // xy
        atomicAdd(&sums[2], dx * dz);  // xz
        atomicAdd(&sums[3], dy * dy);  // yy
        atomicAdd(&sums[4], dy * dz);  // yz
        atomicAdd(&sums[5], dz * dz);  // zz
    }
    __syncthreads();

    // Solve eigensystem on first thread
    if (threadIdx.x == 0) {
        // Form covariance matrix
        float covariance[3][3] = {
            {sums[0], sums[1], sums[2]},
            {sums[1], sums[3], sums[4]},
            {sums[2], sums[4], sums[5]}
        };
        
        // Find smallest eigenvalue and corresponding eigenvector
        // Using power iteration method for simplicity
        float vec[3] = {1.0f, 1.0f, 1.0f};
        float norm;
        
        for (int iter = 0; iter < 10; iter++) {
            float new_vec[3] = {0.0f, 0.0f, 0.0f};
            
            // Matrix multiplication
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    new_vec[i] += covariance[i][j] * vec[j];
                }
            }
            
            // Normalize
            norm = sqrtf(new_vec[0] * new_vec[0] + 
                        new_vec[1] * new_vec[1] + 
                        new_vec[2] * new_vec[2]);
            
            if (norm > 1e-10f) {
                vec[0] = new_vec[0] / norm;
                vec[1] = new_vec[1] / norm;
                vec[2] = new_vec[2] / norm;
            }
        }

        // Set plane coefficients (normal vector and d term)
        plane_coefficients[0] = vec[0];
        plane_coefficients[1] = vec[1];
        plane_coefficients[2] = vec[2];
        plane_coefficients[3] = -(vec[0] * centroid[0] + 
                                 vec[1] * centroid[1] + 
                                 vec[2] * centroid[2]);
    }
}

void cudaGroundSegmentation(float* d_points, int num_points, float* d_coefficients,
                          int* d_indices, float distance_threshold, int max_iterations)
{
    const int BLOCK_SIZE = 256;
    const int num_blocks = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, num_points - 1);

    // RANSAC iterations
    for (int iter = 0; iter < max_iterations; iter++) {
        // Randomly select 3 points for initial plane estimate
        int idx1 = dist(rng);
        int idx2 = dist(rng);
        int idx3 = dist(rng);

        float p1[3] = {d_points[idx1 * 4], d_points[idx1 * 4 + 1], d_points[idx1 * 4 + 2]};
        float p2[3] = {d_points[idx2 * 4], d_points[idx2 * 4 + 1], d_points[idx2 * 4 + 2]};
        float p3[3] = {d_points[idx3 * 4], d_points[idx3 * 4 + 1], d_points[idx3 * 4 + 2]};

        // Calculate initial plane coefficients
        float v1[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
        float v2[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
        
        // Cross product for normal vector
        float normal[3] = {
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        };
        
        // Normalize
        float norm = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        if (norm < 1e-10f) continue;
        
        float temp_coefficients[4] = {
            normal[0] / norm,
            normal[1] / norm,
            normal[2] / norm,
            -(normal[0] * p1[0] + normal[1] * p1[1] + normal[2] * p1[2]) / norm
        };

        // Copy temporary coefficients to device
        cudaMemcpy(d_coefficients, temp_coefficients, 4 * sizeof(float), cudaMemcpyHostToDevice);

        // Find inliers
        findInliersKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_points, num_points, d_coefficients, d_indices, distance_threshold);
        
        // Refine plane using all inliers
        fitPlaneKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_points, d_indices, num_points, d_coefficients);
    }
}