#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */
void checkAccuracy(float *p, int nBodies);
typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

__global__
void bodyForce(Body *p, float dt, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; 
    p[i].vy += dt*Fy; 
    p[i].vz += dt*Fz; 
  }

}
__global__ void add(Body*p, float dt,int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    p[i].x += p[i].vx*dt;
    p[i].y += p[i].vy*dt;
    p[i].z += p[i].vz*dt;
  }
}

int main(const int argc, const char** argv) {
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  /*
   * Do not change the value for `nBodies` here. If you would like to modify it,
   * pass values into the command line.
   */

  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "files/initialized_4096";
    solution_values = "files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "files/initialized_65536";
    solution_values = "files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  cudaMallocManaged(&buf, bytes);
  Body *p = (Body*)buf;
  read_values_from_file(initialized_values, buf, bytes);

  /*
   * As a constraint of this exercise, `randomizeBodies` must remain a host function.
   */
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = 32 * numberOfSMs;

  double totalTime = 0.0;

  /*
   * This simulation will run for 10 cycles of time, calculating gravitational
   * interaction amongst bodies, and adjusting their positions to reflect.
   */

  /*******************************************************************/
  // Do not modify these 2 lines of code.
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
  /*******************************************************************/

  /*
   * You will likely wish to refactor the work being done in `bodyForce`,
   * as well as the work to integrate the positions.
   */
  bodyForce<<< numberOfBlocks, threadsPerBlock >>>(p, dt, nBodies); // compute interbody forces  
  cudaDeviceSynchronize();
  add<<< numberOfBlocks, threadsPerBlock >>>(p, dt, nBodies);
  cudaDeviceSynchronize();
  // Do not modify the code in this section.
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }
  cudaDeviceSynchronize();

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
  /*******************************************************************/

  /*
   * Feel free to modify code below.
   */

  cudaFree(buf);
}