#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>

double wctime() 
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

__global__ void saxpy_par(int n, float a, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		y[i] = a * x[i] + y[i];
	}
}

float saxpy_check(int n, float a, float *x, float *y, float *z)  {
 // a, x, y == original data for saxpy
 // z = result found -- with which to compare.
 float s=0.0, t = 0.0;
 for (int i=0; i<n; i++) {
      y[i] += a * x[i] ;
       s += (y[i] - z[i])*(y[i] - z[i]);
       t += z[i]*z[i];
 }
  if (t == 0.0) return(-1);
    else
  return(sqrt(s/t));
}

int main() {
	//size of vectors
	int n = 8388608; //8*1024*1024
	size_t size = n * sizeof(float);
	
	//allocate vectors on CPU
	float *x , *y, *z;
	x = (float *)malloc(size);
	y = (float *)malloc(size);
	z = (float *)malloc(size);
	
	//allocate vectors on GPU
	//cudaMalloc( void** devPtr, size_t size )
	//cudaSuccess = 0
	float *x_GPU, *y_GPU;
	if (cudaMalloc((void**) &x_GPU, size) != 0) {
		return -1;
	}
	if (cudaMalloc((void**) &y_GPU, size) != 0) {
		return -1;
	}
	float a = 1.0;
	int NITER = 100;
	a = a/(float)NITER;
	
	//Initialize x and y with random numbers
	for (int i = 0; i < n; i++) {
		x[i] = (float)rand()/(float)rand();
		y[i] = (float)rand()/(float)rand(); 
	}

	int vecLen;
	for (vecLen = 2048; vecLen <= n; vecLen*=2) {
		//set grid and block dimensions
		dim3 dimGrid(vecLen/1024);
		dim3 dimBlock(1024);
		//call saxpy_par kernel NITER times
		double t1 = wctime(); //start time
		for (int iter = 0; iter < NITER; iter++) {
			//copy vectors to GPU
			cudaMemcpy(x_GPU, x, vecLen * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(y_GPU, y, vecLen * sizeof(float), cudaMemcpyHostToDevice);
			saxpy_par<<<dimGrid, dimBlock>>>(vecLen, a, x_GPU, y_GPU);
			//Copy result to CPU so it can be passed to saxpy_check
			cudaMemcpy(z, y_GPU, vecLen * sizeof(float), cudaMemcpyDeviceToHost);
		}
		double t2 = wctime(); //end time
		//Check error
		float error = saxpy_check(vecLen, a, x, y, z);
		//get performance stats
		//Perform a multiply and an add for each element in both arrays (2 operations)
		//This happens 
		float flops = (2 * vecLen * NITER)/(t2 - t1);
		printf("** vecLen = %d, Mflops = %.2f, err = %.2e\n", vecLen, flops*1e-6, error);
	}
	free(x);
	free(y);
	free(z);
	
	cudaFree(x_GPU);
	cudaFree(y_GPU);

	return 0;
}
