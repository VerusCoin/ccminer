#include <miner.h>

#include <cuda_helper.h>

typedef uint4 uint128m;
#define GPU_DEBUG
#define VERUS_KEY_SIZE 8832
#define VERUS_KEY_SIZE128 552
#define THREADS 128
#define INNERLOOP 16

#define AES2_EMU(s0, s1, rci) \
  aesenc((unsigned char *)&s0, (unsigned char *)&rc[rci],sharedMemory1); \
  aesenc((unsigned char *)&s1, (unsigned char *)&rc[rci + 1],sharedMemory1); \
  aesenc((unsigned char *)&s0, (unsigned char *)&rc[rci + 2],sharedMemory1); \
  aesenc((unsigned char *)&s1, (unsigned char *)&rc[rci + 3],sharedMemory1);


#define MIX2_EMU(s0, s1) \
  tmp = _mm_unpacklo_epi32_emu(s0, s1); \
  s1 = _mm_unpackhi_epi32_emu(s0, s1); \
  s0 = tmp;

#define AES4(s0, s1, s2, s3, rci) \
  aesenc((unsigned char *)&s0, (unsigned char *)&rc[rci],sharedMemory1); \
  aesenc((unsigned char *)&s1, (unsigned char *)&rc[rci + 1],sharedMemory1); \
  aesenc((unsigned char *)&s2, (unsigned char *)&rc[rci + 2],sharedMemory1); \
  aesenc((unsigned char *)&s3, (unsigned char *)&rc[rci + 3],sharedMemory1); \
  aesenc((unsigned char *)&s0, (unsigned char *)&rc[rci + 4], sharedMemory1); \
  aesenc((unsigned char *)&s1, (unsigned char *)&rc[rci + 5], sharedMemory1); \
  aesenc((unsigned char *)&s2, (unsigned char *)&rc[rci + 6], sharedMemory1); \
  aesenc((unsigned char *)&s3, (unsigned char *)&rc[rci + 7], sharedMemory1);

#define TRUNCSTORE(out, s3) \
  *(uint64_t*)(out + 24) = *(((uint64_t*)&s3 + 0));

#define MIX4(s0, s1, s2, s3) \
  tmp  = _mm_unpacklo_epi32_emu(s0, s1); \
  s0 = _mm_unpackhi_epi32_emu(s0, s1); \
  s1 = _mm_unpacklo_epi32_emu(s2, s3); \
  s2 = _mm_unpackhi_epi32_emu(s2, s3); \
  s3 = _mm_unpacklo_epi32_emu(s0, s2); \
  s0 = _mm_unpackhi_epi32_emu(s0, s2); \
  s2 = _mm_unpackhi_epi32_emu(s1, tmp); \
  s1 = _mm_unpacklo_epi32_emu(s1, tmp);

__host__ void verus_setBlock(uint8_t *blockf, uint32_t *pTargetIn, uint8_t *lkey, int thr_id);


__device__ const uint32_t sbox[] = {
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0,
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0,
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0,
	0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0
};

#define XT(x) (((x) << 1) ^ (((x) >> 7) ? 0x1b : 0))
__global__ void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce);

//__device__ __device__ uint128m local_key[THREADS][VERUS_KEY_SIZE128];
static uint32_t *d_nonces[MAX_GPUS];

__device__ __constant__ uint128m vkey[VERUS_KEY_SIZE128];
__device__ __constant__ uint8_t blockhash_half[64];
__device__ __constant__ uint32_t ptarget[8];

__host__
void verus_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 1 * sizeof(uint32_t)));
	//CUDA_SAFE_CALL(cudaMalloc(&vkey[thr_id], VERUS_KEY_SIZE * sizeof(uint8_t)));
	//CUDA_SAFE_CALL(cudaMalloc(&local_key[thr_id], THREADS * VERUS_KEY_SIZE * sizeof(uint8_t)));
};


__host__
void verus_setBlock(uint8_t *blockf, uint32_t *pTargetIn, uint8_t *lkey, int thr_id)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptarget, (void**)pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half, (void**)blockf, 64 * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(vkey,(void**)lkey, VERUS_KEY_SIZE * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
	

};
__host__
void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	cudaMemset(d_nonces[thr_id], 0xff, 1 * sizeof(uint32_t));
	//CUDA_SAFE_CALL(cudaMemset(local_key[thr_id], 0x00, THREADS * VERUS_KEY_SIZE * sizeof(uint8_t)));
	const uint32_t threadsperblock = THREADS;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	verus_gpu_hash << <grid, block >> >(threads, startNonce, d_nonces[thr_id]);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_nonces[thr_id], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	//memcpy(resNonces, h_nonces[thr_id], NBN * sizeof(uint32_t));

};

__device__  __forceinline__ uint128m _mm_clmulepi64_si128_emu(uint128m ai, uint128m bi, int imm)
{
	uint64_t a = *((uint64_t*)&ai + (imm & 1));

	uint64_t b = *((uint64_t*)&bi + ((imm & 0x10) >> 4));
	
	uint8_t  i; //window size s = 4,
	//uint64_t two_s = 16; //2^s
	//uint64_t smask = 15; //s 15
	uint64_t u[16];
	uint64_t r[2];
	uint64_t tmp;
	uint64_t ifmask;
	//Precomputation
	u[0] = 0;
	u[1] = b;
#pragma unroll
	for (i = 2; i < 16; i += 2) {
		u[i] = u[i >> 1] << 1; //even indices: left shift
		u[i + 1] = u[i] ^ b; //odd indices: xor b
	}
	//Multiply
	r[0] = u[a & 15]; //first window only affects lower word
	r[1] = 0;
#pragma unroll
	for (i = 4; i < 64; i += 4) {
		tmp = u[a >> i & 15];
		r[0] ^= tmp << i;
		r[1] ^= tmp >> (64 - i);
	}
	//Repair
	uint64_t m = 0xEEEEEEEEEEEEEEEE; //s=4 => 16 times 1110
#pragma unroll
	for (i = 1; i < 4; i++) {
		tmp = ((a & m) >> i);
		m &= m << 1; //shift mask to exclude all bit j': j' mod s = i
		ifmask = -((b >> (64 - i)) & 1); //if the (64-i)th bit of b is 1
		r[1] ^= (tmp & ifmask);
	}
	uint128m out;
	((uint64_t*)&out)[0] = r[0];
	((uint64_t*)&out)[1] = r[1];
	return out;
}

__device__   __forceinline__ void aesenc(unsigned char *s, const unsigned char *rk, uint32_t *sharedMemory1)
{
	//uint32_t  t, u, w;
	//uint32_t v[4][4];

#define XT4(x) ((((x) << 1) & 0xfefefefe) ^ ((((x) >> 31) & 1) ? 0x1b000000 : 0)^ ((((x) >> 23)&1) ? 0x001b0000 : 0)^ ((((x) >> 15)&1) ? 0x00001b00 : 0)^ ((((x) >> 7)&1) ? 0x0000001b : 0))

	uint32_t  t, u;
	uint32_t v[4];

	((uint8_t*)&v[0])[0] = ((uint8_t*)&sharedMemory1[0])[s[0]];
	((uint8_t*)&v[0])[7] = ((uint8_t*)&sharedMemory1[0])[s[1]];
	((uint8_t*)&v[0])[10] = ((uint8_t*)&sharedMemory1[0])[s[2]];
	((uint8_t*)&v[0])[13] = ((uint8_t*)&sharedMemory1[0])[s[3]];
	((uint8_t*)&v[0])[1] = ((uint8_t*)&sharedMemory1[0])[s[4]];
	((uint8_t*)&v[0])[4] = ((uint8_t*)&sharedMemory1[0])[s[5]];
	((uint8_t*)&v[0])[11] = ((uint8_t*)&sharedMemory1[0])[s[6]];
	((uint8_t*)&v[0])[14] = ((uint8_t*)&sharedMemory1[0])[s[7]];
	((uint8_t*)&v[0])[2] = ((uint8_t*)&sharedMemory1[0])[s[8]];
	((uint8_t*)&v[0])[5] = ((uint8_t*)&sharedMemory1[0])[s[9]];
	((uint8_t*)&v[0])[8] = ((uint8_t*)&sharedMemory1[0])[s[10]];
	((uint8_t*)&v[0])[15] = ((uint8_t*)&sharedMemory1[0])[s[11]];
	((uint8_t*)&v[0])[3] = ((uint8_t*)&sharedMemory1[0])[s[12]];
	((uint8_t*)&v[0])[6] = ((uint8_t*)&sharedMemory1[0])[s[13]];
	((uint8_t*)&v[0])[9] = ((uint8_t*)&sharedMemory1[0])[s[14]];
	((uint8_t*)&v[0])[12] = ((uint8_t*)&sharedMemory1[0])[s[15]];

	t = v[0];
	u = v[0] ^ v[1] ^ v[2] ^ v[3];
	v[0] = v[0] ^ u ^ XT4(v[0] ^ v[1]);
	v[1] = v[1] ^ u ^ XT4(v[1] ^ v[2]);
	v[2] = v[2] ^ u ^ XT4(v[2] ^ v[3]);
	v[3] = v[3] ^ u ^ XT4(v[3] ^ t);

	s[0] = ((uint8_t*)&v[0])[0] ^ rk[0];
	s[1] = ((uint8_t*)&v[0])[4] ^ rk[1];
	s[2] = ((uint8_t*)&v[0])[8] ^ rk[2];
	s[3] = ((uint8_t*)&v[0])[12] ^ rk[3];
	s[4] = ((uint8_t*)&v[0])[1] ^ rk[4];
	s[5] = ((uint8_t*)&v[0])[5] ^ rk[5];
	
	s[6] = ((uint8_t*)&v[0])[9] ^ rk[6];
	s[7] = ((uint8_t*)&v[0])[13] ^ rk[7];
	s[8] = ((uint8_t*)&v[0])[2] ^ rk[8];
	s[9] = ((uint8_t*)&v[0])[6] ^ rk[9];
	s[10] = ((uint8_t*)&v[0])[10] ^ rk[10];
	s[11] = ((uint8_t*)&v[0])[14] ^ rk[11];
	s[12] = ((uint8_t*)&v[0])[3] ^ rk[12];
	s[13] = ((uint8_t*)&v[0])[7] ^ rk[13];
	s[14] = ((uint8_t*)&v[0])[11] ^ rk[14];
	s[15] = ((uint8_t*)&v[0])[15] ^ rk[15];

}

#define _mm_xor_si128_emu(a,b) (operator^(a,b))

#define _mm_load_si128_emu(p) (*(uint128m*)(p));

#define _mm_cvtsi128_si64_emu(p) (((int64_t *)&p)[0]);

#define _mm_cvtsi128_si32_emu(p) (((int32_t *)&a)[0]);

__device__  __forceinline__ uint128m _mm_cvtsi32_si128_emu(uint32_t lo)
{
	uint128m result;
	((uint32_t *)&result)[0] = lo;
	((uint32_t *)&result)[1] = 0;
	((uint64_t *)&result)[1] = 0;
	return result;
}
__device__  __forceinline__ uint128m _mm_cvtsi64_si128_emu(uint64_t lo)
{
	uint128m result;
	((uint64_t *)&result)[0] = lo;
	((uint64_t *)&result)[1] = 0;
	return result;
}
__device__  __forceinline__ uint128m _mm_set_epi64x_emu(uint64_t hi, uint64_t lo)
{
	uint128m result;
	((uint64_t *)&result)[0] = lo;
	((uint64_t *)&result)[1] = hi;
	return result;
}
__device__  __forceinline__ uint128m _mm_shuffle_epi8_emu(uint128m a, uint128m b)
{
	uint128m result;
	for (int i = 0; i < 16; i++)
	{
		if (((uint8_t *)&b)[i] & 0x80)
		{
			((uint8_t *)&result)[i] = 0;
		}
		else
		{
			((uint8_t *)&result)[i] = ((uint8_t *)&a)[((uint8_t *)&b)[i] & 0xf];
		}
	}

	return result;
}

__device__  __forceinline__ uint128m _mm_setr_epi8_emu(u_char c0, u_char c1, u_char c2, u_char c3, u_char c4, u_char c5, u_char c6, u_char c7, u_char c8, u_char c9, u_char c10, u_char c11, u_char c12, u_char c13, u_char c14, u_char c15)
{

		uint128m result;

	
		((uint32_t *)&result)[0] = 0x2d361b00;
		((uint32_t *)&result)[1] = 0x415a776c;
		((uint32_t *)&result)[2] = 0xf5eec3d8;
		((uint32_t *)&result)[3] = 0x9982afb4;

	return result;
}


__device__  __forceinline__ uint128m _mm_srli_si128_emu(uint128m input, int imm8)
{
	//we can cheat here as its an 8 byte shift just copy the 64bits
	uint128m temp;
	((uint64_t*)&temp)[0] = ((uint64_t*)&input)[1];
	((uint64_t*)&temp)[1] = 0;


	return temp;
}
__device__  __forceinline__ uint128m _mm_unpacklo_epi32_emu(uint128m a, uint128m b)
{
	uint32_t result[4];
	uint32_t *tmp1 = (uint32_t *)&a, *tmp2 = (uint32_t *)&b;
	result[0] = tmp1[0];
	result[1] = tmp2[0];
	result[2] = tmp1[1];
	result[3] = tmp2[1];
	return ((uint128m *)&result[0])[0];
}

__device__  __forceinline__ uint128m _mm_unpackhi_epi32_emu(uint128m a, uint128m b)
{
	uint32_t result[4];
	uint32_t *tmp1 = (uint32_t *)&a, *tmp2 = (uint32_t *)&b;
	result[0] = tmp1[2];
	result[1] = tmp2[2];
	result[2] = tmp1[3];
	result[3] = tmp2[3];
	return ((uint128m *)&result[0])[0];
}


__device__ __forceinline__ uint128m _mm_mulhrs_epi16_emu(uint128m _a, uint128m _b)
{
	int16_t result[8];
	int16_t *a = (int16_t*)&_a, *b = (int16_t*)&_b;
#pragma unroll 8
	for (int i = 0; i < 8; i++)
	{
		result[i] = (int16_t)((((int32_t)(a[i]) * (int32_t)(b[i])) + 0x4000) >> 15);
	}
	return *(uint128m *)result;
}


__device__ uint128m __verusclmulwithoutreduction64alignedrepeatgpu(uint128m *randomsource, const  uint128m buf [4], uint64_t keyMask, uint32_t *sharedMemory1)
{
    uint128m const *pbuf;
	keyMask >>= 4;
	uint128m acc = randomsource[keyMask + 2];
	
#ifdef GPU_DEBUGGY
	if (nounce == 0)
	{
		printf("[GPU]BUF ito verusclmulithout        : ");
		for (int i = 0; i < 64; i++)
			printf("%02x", ((uint8_t*)&buf[0])[i]);
		printf("\n");
		printf("[GPU]KEy ito verusclmulithout        : ");
		for (int e = 0; e < 64; e++)
		printf("%02x", ((uint8_t*)&randomsource[0])[e]);
	printf("\n");
	    printf("[GPU]ACC ito verusclmulithout        : ");
	for (int i = 0; i < 16; i++)
		printf("%02x", ((uint8_t*)&acc)[i]);
	printf("\n");
	}
#endif	
	// divide key mask by 32 from bytes to uint128m
	
	uint32_t prand_idx, prandex_idx;
	uint64_t selector;
	uint128m prand;
	uint128m prandex;

	for (int64_t i = 0; i < 32; i++)
	{
		
		selector = _mm_cvtsi128_si64_emu(acc);

		
		prand_idx = ((selector >> 5) & keyMask);
		prandex_idx = ((selector >> 32) & keyMask);
		// get two random locations in the key, which will be mutated and swapped
		
		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];

	//	save_rand[i] = ((selector >> 5) & keyMask);
	//	save_randex[i] = ((selector >> 32) & keyMask);

		// select random start and order of pbuf processing
		pbuf = buf + (selector & 3);
		uint64_t case_v;
		case_v = selector &  0x1cu;
#ifdef GPU_DEBUGu
		uint64_t egg, nog, salad;
		if (nounce == 0)
		{
			printf("[GPU]*****LOOP[%d]**********\n",i);
			egg = selector & 0x03u;
			nog = ((selector >> 32) & keyMask);
			salad = ((selector >> 5) & keyMask);
			printf("[GPU]selector: %llx\n case: %llx selector &3: ", selector, case_v);
			printf("%llx \n", egg);
			printf("[GPU]((selector >> 32) & keyMask) %d",nog);
			printf("[GPU]((selector >> 5) & keyMask) %d", salad);
			printf("\nacc     : ");
			printf("%016llx%016llx", ((uint64_t*)&acc)[0], ((uint64_t*)&acc)[1]);
			printf("\n");

			printf("[GPU]prand   : ");
			//for (int e = 0; e < 4; e++)
			printf("%016llx%016llx", ((uint64_t*)&prand)[0], ((uint64_t*)&prand)[1]);
			printf("\n");
			printf("[GPU]prandex : ");
			//for (int e = 0; e < 16; e++)
			printf("%016llx%016llx", ((uint64_t*)&prandex)[0], ((uint64_t*)&prandex)[1]);
			printf("\n");


		}

#endif
		
		if((case_v) == 0)
		{
		const uint128m temp1 = prandex;
	
			const uint128m temp2 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			

			const uint128m add1 = _mm_xor_si128_emu(temp1, temp2);

			const uint128m clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);

			const uint128m tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const uint128m tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			const uint128m temp12 = prand;
			prand = tempa2;


			const uint128m temp22 = _mm_load_si128_emu(pbuf);
			const uint128m add12 = _mm_xor_si128_emu(temp12, temp22);
			const uint128m clprod12 = _mm_clmulepi64_si128_emu(add12, add12, 0x10);
			acc = _mm_xor_si128_emu(clprod12, acc);

			const uint128m tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const uint128m tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			prandex = tempb2;

		
			
		}
		if (case_v == 4)
		{
			const uint128m temp1 = prand;
			const uint128m temp2 = _mm_load_si128_emu(pbuf);
			const uint128m add1 = _mm_xor_si128_emu(temp1, temp2);
			const uint128m clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);
			const uint128m clprod2 = _mm_clmulepi64_si128_emu(temp2, temp2, 0x10);
			acc = _mm_xor_si128_emu(clprod2, acc);

			const uint128m tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const uint128m tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			const uint128m temp12 = prandex;
			prandex= tempa2;

			const uint128m temp22 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const uint128m add12 = _mm_xor_si128_emu(temp12, temp22);
			acc = _mm_xor_si128_emu(add12, acc);

			const uint128m tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const uint128m tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			prand= tempb2;
	
		}
		if (case_v == 8)
		{
			const uint128m temp1 = prandex;
			const uint128m temp2 = _mm_load_si128_emu(pbuf);
			const uint128m add1 = _mm_xor_si128_emu(temp1, temp2);
			acc = _mm_xor_si128_emu(add1, acc);

			const uint128m tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const uint128m tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			const uint128m temp12 = prand;
			prand= tempa2;

			const uint128m temp22 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const uint128m add12 = _mm_xor_si128_emu(temp12, temp22);
			const uint128m clprod12 = _mm_clmulepi64_si128_emu(add12, add12, 0x10);
			acc = _mm_xor_si128_emu(clprod12, acc);
			const uint128m clprod22 = _mm_clmulepi64_si128_emu(temp22, temp22, 0x10);
			acc = _mm_xor_si128_emu(clprod22, acc);

			const uint128m tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const uint128m tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			prandex=tempb2;
			
		}
		if (case_v == 0xc)
		{
			const uint128m temp1 = prand;
			const uint128m temp2 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const uint128m add1 = _mm_xor_si128_emu(temp1, temp2);

			// cannot be zero here
			const int32_t divisor = ((uint32_t*)&selector)[0];

			acc = _mm_xor_si128_emu(add1, acc);

			int64_t dividend = _mm_cvtsi128_si64_emu(acc);
			int64_t tmpmod = dividend % divisor;
			const uint128m modulo = _mm_cvtsi32_si128_emu(tmpmod);
			acc = _mm_xor_si128_emu(modulo, acc);

			const uint128m tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const uint128m tempa2 = _mm_xor_si128_emu(tempa1, temp1);
			dividend &= 1;
			if (dividend)
			{
				const uint128m temp12 = prandex;
				prandex = tempa2;

				const uint128m temp22 = _mm_load_si128_emu(pbuf);
				const uint128m add12 = _mm_xor_si128_emu(temp12, temp22);
				const uint128m clprod12 = _mm_clmulepi64_si128_emu(add12, add12, 0x10);
				acc = _mm_xor_si128_emu(clprod12, acc);
				const uint128m clprod22 = _mm_clmulepi64_si128_emu(temp22, temp22, 0x10);
				acc = _mm_xor_si128_emu(clprod22, acc);

				const uint128m tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
				const uint128m tempb2 = _mm_xor_si128_emu(tempb1, temp12);
				prand = tempb2;
			}
			else
			{
				const uint128m tempb3 = prandex;
				prandex = tempa2;
				prand = tempb3;
			}

		}
		if (case_v == 0x10)
		{
			// a few AES operations
			uint128m rc[12];
			
			rc[0] = prand; 

			rc[1] = randomsource[prand_idx + 1];
			rc[2] = randomsource[prand_idx + 2];
			rc[3] = randomsource[prand_idx + 3];
			rc[4] = randomsource[prand_idx + 4];
			rc[5] = randomsource[prand_idx + 5];
			rc[6] = randomsource[prand_idx + 6];
			rc[7] = randomsource[prand_idx + 7];
			rc[8] = randomsource[prand_idx + 8];
			rc[9] = randomsource[prand_idx + 9];
			rc[10] = randomsource[prand_idx + 10];
			rc[11] = randomsource[prand_idx + 11];
			uint128m tmp;
			const uint64_t rr = 0;
			uint128m temp1 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			uint128m temp2 = _mm_load_si128_emu(pbuf);
			
			AES2_EMU(temp1, temp2, 0);
			MIX2_EMU(temp1, temp2);


			AES2_EMU(temp1, temp2, 4);
			MIX2_EMU(temp1, temp2);

			AES2_EMU(temp1, temp2, 8);
			MIX2_EMU(temp1, temp2);


		    acc = _mm_xor_si128_emu(temp1, acc);
			acc = _mm_xor_si128_emu(temp2, acc);

			const uint128m tempa1 = prand;
			const uint128m tempa2 = _mm_mulhrs_epi16_emu(acc, tempa1);
			const uint128m tempa3 = _mm_xor_si128_emu(tempa1, tempa2);

			const uint128m tempa4 = prandex;
			prandex = tempa3;
			prand = tempa4;


		}
		if(case_v == 0x14)
		{
			// we'll just call this one the monkins loop, inspired by Chris
			const uint128m *buftmp = pbuf - (((selector & 1) << 1) - 1);
			uint128m tmp; // used by MIX2

			uint64_t rounds = selector >> 61; // loop randomly between 1 and 8 times
			uint128m *rc = &randomsource[prand_idx];


			uint64_t aesround = 0;
			uint128m onekey;
			uint64_t loop_c; 
		
			do
			{
				loop_c = selector & (0x10000000 << rounds);
				if (loop_c)
				{
					onekey = _mm_load_si128_emu(rc++);
					const uint128m temp2 = _mm_load_si128_emu(rounds & 1 ? pbuf : buftmp);
					const uint128m add1 = _mm_xor_si128_emu(onekey, temp2);
					const uint128m clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
					acc = _mm_xor_si128_emu(clprod1, acc);
				}
				else
				{
					onekey = _mm_load_si128_emu(rc++);
					uint128m temp2 = _mm_load_si128_emu(rounds & 1 ? buftmp : pbuf);
				
					const uint64_t roundidx = aesround++ << 2;
					AES2_EMU(onekey, temp2, roundidx);
				
					MIX2_EMU(onekey, temp2);
				
					acc = _mm_xor_si128_emu(onekey, acc);
					acc = _mm_xor_si128_emu(temp2, acc);

				}

			} while (rounds--);

			const uint128m tempa1 = (prand);
			const uint128m tempa2 = _mm_mulhrs_epi16_emu(acc, tempa1);
			const uint128m tempa3 = _mm_xor_si128_emu(tempa1, tempa2);

			const uint128m tempa4 = (prandex);
			prandex = tempa3;
			prand = tempa4;

		}
		if(case_v == 0x18)
		{
			const uint128m temp1 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const uint128m temp2 = (prand);
			const uint128m add1 = _mm_xor_si128_emu(temp1, temp2);
			const uint128m clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);

			const uint128m tempa1 = _mm_mulhrs_epi16_emu(acc, temp2);
			const uint128m tempa2 = _mm_xor_si128_emu(tempa1, temp2);

			const uint128m tempb3 = (prandex);
			prandex = tempa2;
			prand = tempb3;
			
		}
		if(case_v == 0x1c)
		{
			const uint128m temp1 = _mm_load_si128_emu(pbuf);
			const uint128m temp2 = (prandex);
			const uint128m add1 = _mm_xor_si128_emu(temp1, temp2);
			const uint128m clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);


			const uint128m tempa1 = _mm_mulhrs_epi16_emu(acc, temp2);
			const uint128m tempa2 = _mm_xor_si128_emu(tempa1, temp2);
			const uint128m tempa3 = (prand);

			
			prand = tempa2;

			acc = _mm_xor_si128_emu(tempa3, acc);

			const uint128m tempb1 = _mm_mulhrs_epi16_emu(acc, tempa3);
			const uint128m tempb2 = _mm_xor_si128_emu(tempb1, tempa3);
			prandex = tempb2;

		}	

		 randomsource[prand_idx] = prand;
		 randomsource[prandex_idx] = prandex;

	}

	return acc;
}


__device__   __forceinline__ void haraka512_port_keyed2222(unsigned char *out, const unsigned char *in, uint128m *rc, uint32_t *sharedMemory1, uint32_t nonce)
{
	uint128m s[4], tmp;

	s[0] = ((uint128m*)&in[0])[0];
	s[1] = ((uint128m*)&in[0])[1];
	s[2] = ((uint128m*)&in[0])[2];
	s[3] = ((uint128m*)&in[0])[3];

	AES4(s[0], s[1], s[2], s[3], 0);
	MIX4(s[0], s[1], s[2], s[3]);

	AES4(s[0], s[1], s[2], s[3], 8);
	MIX4(s[0], s[1], s[2], s[3]);

	AES4(s[0], s[1], s[2], s[3], 16);
	MIX4(s[0], s[1], s[2], s[3]);

	AES4(s[0], s[1], s[2], s[3], 24);
	MIX4(s[0], s[1], s[2], s[3]);

	AES4(s[0], s[1], s[2], s[3], 32);
	MIX4(s[0], s[1], s[2], s[3]);

	//s[0] = _mm_xor_si128_emu(s[0], ((uint128m*)&in[0])[0]);
	//s[1] = _mm_xor_si128_emu(s[1], ((uint128m*)&in[0])[1]);
	//s[2] = _mm_xor_si128_emu(s[2], ((uint128m*)&in[0])[2]);
	s[3] = _mm_xor_si128_emu(s[3], ((uint128m*)&in[0])[3]);

	TRUNCSTORE(out, s[3]);

	//((uint32_t*)&out[0])[7] = ((uint32_t*)&s[52])[0] ^ ((uint32_t*)&in[52])[0];

}

__device__   __forceinline__ uint128m precompReduction64_si128(uint128m A) {

	//const uint128m C = _mm_set_epi64x(1U,(1U<<4)+(1U<<3)+(1U<<1)+(1U<<0)); // C is the irreducible poly. (64,4,3,1,0)
	//const uint128m C = _mm_cvtsi64_si128_emu(27U);
	uint128m M;
	M.x = 0x2d361b00;
	M.y = 0x415a776c;
	M.z = 0xf5eec3d8;
	M.w = 0x9982afb4;


	uint128m Q2 = _mm_clmulepi64_si128_emu(A, _mm_cvtsi64_si128_emu(27U), 0x01);
	uint128m Q3 = _mm_shuffle_epi8_emu(M,_mm_srli_si128_emu(Q2, 8));

	uint128m Q4 = _mm_xor_si128_emu(Q2, A);
	const uint128m final = _mm_xor_si128_emu(Q3, Q4);
	return final;/// WARNING: HIGH 64 BITS SHOULD BE ASSUMED TO CONTAIN GARBAGE
}

__device__  __forceinline__ uint64_t precompReduction64(uint128m A) {
	uint128m tmp = precompReduction64_si128(A);
	return _mm_cvtsi128_si64_emu(tmp);
}

__global__ __launch_bounds__(THREADS, 2)
void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint128m mid, biddy[VERUS_KEY_SIZE128];
	int i;
	uint8_t s[64];
	uint32_t nounce = startNonce + thread, hash[32] = { 0 };
	uint64_t acc;
	__shared__ uint32_t sharedMemory1[THREADS];

	
	//uint32_t save_rand[32] = { 0 };
	//uint32_t save_randex[32] = { 0 };
	
	memcpy(s, blockhash_half, 64);
	memcpy(s + 47, blockhash_half, 16);
	memcpy(s + 63, blockhash_half, 1);
//	if (blockIdx.x < 10)
	memcpy(biddy, vkey, VERUS_KEY_SIZE); // 2% speed increase


	sharedMemory1[threadIdx.x] = sbox[threadIdx.x];// copy sbox to shared mem
	

	((uint32_t *)&s)[8] = nounce;
	
	uint128m lazy;
	((uint64_t *)&lazy)[0] = 0x0000000000010000ull;
	((uint64_t *)&lazy)[1] = 0x0000000000000000ull;
	__syncthreads();
	mid = __verusclmulwithoutreduction64alignedrepeatgpu(biddy, (uint128m*)s, 8191, sharedMemory1);

	mid = _mm_xor_si128_emu(mid, lazy);

	
	acc = precompReduction64(mid);

	memcpy(s + 47, &acc, 8);
	memcpy(s + 55, &acc, 8);
	memcpy(s + 63, &acc, 1);
	uint64_t mask = 8191 >> 4;
	mask &= acc;
	
	//haraka512_port_keyed((unsigned char*)hash, (const unsigned char*)s, (const unsigned char*)(biddy + mask), sharedMemory1, nounce);

	haraka512_port_keyed2222((unsigned char*)hash, (const unsigned char*)s, (biddy + mask), sharedMemory1,nounce);

	if (hash[7] < ptarget[7]) { 
		
		resNonce[0] = nounce;

	//	printf("[GPU]Final hash    : ");//	for (int i = 0; i < 32; i++)//		printf("%02x", ((uint8_t*)&hash[0])[i]);
//printf("\n");
	}

	//__syncthreads();
};
