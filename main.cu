#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>

#define INPUT_SIZE 55
#define RANDOM_SIZE 15

const char base64[65] = "+/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
__device__ __constant__ char cuda_base64[65] = "+/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// perform sha256 calculation here
__global__ void sha256_cuda(JOB ** jobs, int n, int len_prefix) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(thread_id > n) {
    return;
  }
  
  SHA256_CTX ctx;  
  uint8_t data[64], digest[32], i, j;
  uint64_t counter = 0;
  memcpy(data, jobs[thread_id]->data, 64);
  memcpy(digest, jobs[thread_id]->digest, 32);
  
  for(; ; counter++) {

    if(counter % 8192 == 0) {
      jobs[thread_id]->counter = counter;
    }

    for (i = len_prefix; ; i++) {
      
      if (data[i] != 'z') {
	j = counter >> (6 * (i - len_prefix));
	data[i] = cuda_base64[j % 64];
	break;
      }
      data[i] = '+';
    }
    
    sha256_init(&ctx);
    sha256_transform(&ctx, data);

    if(ctx.state[0] != 0) continue;
    sha256_final(&ctx, digest);
    
    for(i = 8; ; i++) {
      
      if(digest[i] < jobs[thread_id]->best_digest[i]) {
	memcpy(jobs[thread_id]->best_digest, digest, 32);
	memcpy(jobs[thread_id]->best_data, data, 55);
	break;
      }
      else if(digest[i] > jobs[thread_id]->best_digest[i]) {
	break;
      }
    }
  }
}

void runJobs(JOB ** jobs, int n, int len_prefix){
  int blockSize;
  int minGridSize;
  int gridSize;
  uint64_t counter = 0;

  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sha256_cuda, 0, 0);
  gridSize = (n + blockSize - 1) / blockSize; 
  
  sha256_cuda <<< gridSize, blockSize >>> (jobs, n, len_prefix);

  for(; ; ) {
    sleep(5);
    printf("\e[1;1H\e[2J");
    print_status(jobs, n, &counter, 5);
  }
  
}


JOB * JOB_init(char * input) {
  JOB * j;
  checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
  memcpy(j->data, input, INPUT_SIZE);

  // finalize input string
  j->data[55] = 128;
  j->data[56] = 0;
  j->data[57] = 0;
  j->data[58] = 0;
  j->data[59] = 0;
  j->data[60] = 0;
  j->data[61] = 0;
  j->data[62] = 1;
  j->data[63] = 184;
  
  memset(j->digest, 0xff, 32);
  memset(j->best_digest, 0xff, 32);
  
  return j;
}

// initialize string to be used for hashing
// input = prefix + '/' + '+' * (INPUT_SIZE - RANDOM_SIZE - strlen(input)) + random_string[RANDOM_SIZE]
void string_init(char * prefix, char * input, unsigned int * seed) {
  
  for(int i = 0; i < strlen(prefix); i ++) {
    input[i] = prefix[i];
  }
  input[strlen(prefix)] = '/';
  
  for(int i = strlen(input); i < INPUT_SIZE - RANDOM_SIZE; i++) {
    input[i] = '+';
  }
  
  for(int i = strlen(input); i < INPUT_SIZE; i++) {
    input[i] = base64[rand_r(seed) % strlen(base64)];
  }
  
}

int main(int argc, char **argv) {
  int n = 65536; // number of jobs
  JOB ** jobs;
  char * prefix, input[INPUT_SIZE] = {};
  unsigned int seed = time(0);
  
  if(argc != 2) {
    fprintf(stderr, "usage: cudasha256 name\n");
    return 1;
  }
  
  prefix = argv[1]; // prefix == name
  checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));
  
  for(int i = 0; i < n; i++) {
    string_init(prefix, input, &seed);
    jobs[i] = JOB_init(input);
    memset(input, 0, INPUT_SIZE);
  }
  
  runJobs(jobs, n, strlen(prefix)+1);
  
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
