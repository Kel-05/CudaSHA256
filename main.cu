#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include <dirent.h>
#include <ctype.h>

#define INPUT_SIZE 55
#define RANDOM_SIZE 15

const char base64[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/+";

__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, INPUT_SIZE);
		sha256_final(&ctx, jobs[i]->digest);
	}
}


void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


JOB * JOB_init(char * input) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	memcpy(j->data, input, INPUT_SIZE);
	
	for (int i = 0; i < 32; i++) {
		j->digest[i] = 0xff;
	}

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
	int n; // number of jobs
	JOB ** jobs;
	char * prefix, input[INPUT_SIZE] = {};
	unsigned int seed = time(0);

	if(argc != 3) {
		fprintf(stderr, "usage: cudasha256 name jobs\n");
		return 1;
	}

	prefix = argv[1]; // prefix == name
	n = atoi(argv[2]);
	checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

	for(int i = 0; i < n; i++) {
		string_init(prefix, input, &seed);
		jobs[i] = JOB_init(input);
		memset(input, 0, INPUT_SIZE);
	}

	runJobs(jobs, n);

	cudaDeviceSynchronize();
	print_jobs(jobs, n);
	cudaDeviceReset();
	return 0;
}
