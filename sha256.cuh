#ifndef SHA256_H
#define SHA256_H


/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
        printf("GPU: cudaError %d (%s)\n", err, cudaGetErrorString(err)); \
}
/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef uint32_t  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct JOB {
  BYTE data[64];
  BYTE digest[32];
  BYTE best_data[64];
  BYTE best_digest[32];
  uint64_t counter = 0;
}JOB;


typedef struct {
  WORD state[8];
} SHA256_CTX;

__device__ __constant__ WORD dev_k[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DECLARATIONS **********************/
__device__ void sha256_init(SHA256_CTX *ctx);
__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[]);


char * hash_to_string(BYTE * buff) {
  char * string = (char *)malloc(70);
  int k, i;
  for (i = 0, k = 0; i < 12; i++, k+= 2)
    {
      sprintf(string + k, "%.2x", buff[i]);
      //printf("%02x", buff[i]);
    }
  string[64] = 0;
  return string;
}

uint64_t print_best_job(JOB ** jobs, int n) {
  uint64_t total_counter = 0;
  int bestjob = 0;
  
  for (int i = 1; i < n; i++) {
    total_counter += jobs[i]->counter;
    for (int j=0; j < 32; j++) {
      
      if (jobs[i]->best_digest[j] < jobs[bestjob]->best_digest[j]) {
	bestjob = i;
	break;
      }
      else if (jobs[i]->best_digest[j] > jobs[bestjob]->best_digest[j]) {
	break;
      }
      
    }
  }
  printf("%s -> %s...\n", jobs[bestjob]->best_data, hash_to_string(jobs[bestjob]->best_digest));
  printf("counter = %lu\n\n", total_counter);
  return total_counter;
}

void print_status(JOB ** jobs, int n, uint64_t * previous_counter, int time) {
  uint64_t total_counter;
  
  printf("\nBest hash: ");
  total_counter = print_best_job(jobs, n);
  printf("speed = %d MiH/s\n", ((total_counter - *previous_counter) / time) >> 20);
  *previous_counter = total_counter;
}

__device__ void mycpy12(uint32_t *d, const uint32_t *s) {
#pragma unroll 3
  for (int k=0; k < 3; k++) d[k] = s[k];
}

__device__ void mycpy16(uint32_t *d, const uint32_t *s) {
#pragma unroll 4
  for (int k=0; k < 4; k++) d[k] = s[k];
}

__device__ void mycpy32(uint32_t *d, const uint32_t *s) {
#pragma unroll 8
  for (int k=0; k < 8; k++) d[k] = s[k];
}

__device__ void mycpy44(uint32_t *d, const uint32_t *s) {
#pragma unroll 11
  for (int k=0; k < 11; k++) d[k] = s[k];
}

__device__ void mycpy48(uint32_t *d, const uint32_t *s) {
#pragma unroll 12
  for (int k=0; k < 12; k++) d[k] = s[k];
}

__device__ void mycpy64(uint32_t *d, const uint32_t *s) {
#pragma unroll 16
  for (int k=0; k < 16; k++) d[k] = s[k];
}

__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
  WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
  
#pragma unroll 16
  for (i = 0, j = 0; i < 16; ++i, j += 4)
    m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
  
#pragma unroll 64
  for (; i < 64; ++i)
    m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
  
  a = ctx->state[0];
  b = ctx->state[1];
  c = ctx->state[2];
  d = ctx->state[3];
  e = ctx->state[4];
  f = ctx->state[5];
  g = ctx->state[6];
  h = ctx->state[7];
  
#pragma unroll 64
  for (i = 0; i < 64; ++i) {
    t1 = h + EP1(e) + CH(e, f, g) + dev_k[i] + m[i];
    t2 = EP0(a) + MAJ(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  
  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

__device__ void sha256_init(SHA256_CTX *ctx)
{
  ctx->state[0] = 0x6a09e667;
  ctx->state[1] = 0xbb67ae85;
  ctx->state[2] = 0x3c6ef372;
  ctx->state[3] = 0xa54ff53a;
  ctx->state[4] = 0x510e527f;
  ctx->state[5] = 0x9b05688c;
  ctx->state[6] = 0x1f83d9ab;
  ctx->state[7] = 0x5be0cd19;
}

__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
  WORD i;

  // Since this implementation uses little endian byte ordering and SHA uses big endian,
  // reverse all the bytes when copying the final state to the output hash.
  for (i = 0; i < 4; ++i) {
    hash[i] = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 4] = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
    hash[i + 8] = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
  }
}

#endif   // SHA256_H
