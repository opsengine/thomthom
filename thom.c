/**
 * Copyright (c) 2010, by:      Angelo Marletta <marlonx80@hotmail.com>
 *
 * This file may be used subject to the terms and conditions of the
 * GNU Library General Public License Version 2, or any later version
 * at your option, as published by the Free Software Foundation.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Library General Public License for more details.
 *
 *******************************************************************************
 *
 * This program is an insanely optimized simulator for solving the Thomson Problem.
 * In the Thomson Problem, N particles are free to move on the surface of a sphere
 * and are repulsing one each other, with a force proportional to the inverse ot their square distance
 * The simulator with an evolutive approach attempts to find a stable configuration of particles
 * in which the resulting force on every particle is almost zero, and the potential minimum.
 * The algorithm explained:
 *    The simulation is composed of a number of sequential cycles, which is fixed (CYCLES)
 *    During one cycle the following things happen:
 *    - For each particle, calculate the sum of the forces applied by other particles
 *    - Get the tangential component to the sphere of this resulting force
 *    - Apply the force to each particle, adding a fraction (Kr) of the force to the particle coordinates
 *    - Normalize the coordinates in order to make the particles stay exactly on the surface of the sphere (norm=1)
 *    - Save a fraction (Ka) of the applied forces, in order simulate inertia and friction
 *      Inertia serves to speed up the simulation and to make the evolution more realistic
 *      Friction serves to loose potential energy and reach a stable state
 *
 * The algorithm complexity is O(N^2)
 *
 * You shoud have a SSE-enabled cpu in order to get the best performance
 *
 * Author: Angelo Marletta
 * Email: marlonx80@hotmail.com
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <errno.h>

#ifdef WIN32
#include <windows.h>
#endif

//number of particles
#define N 60
//repulsive force constant
#define Kr 1e-3
//friction force constant (0-> static, 1->friction-less)
#define Ka 0.5
//maximum number of simulation cycles
#define CYCLES 10000
#define PRINT_INTERVAL 1000
// #define DEBUG
#define OPTIMIZE_NORMALIZATION
#define OPTIMIZE_SSE

//TODO: add SSE3 haddps instruction where can be applied
// #define USE_SSE3

#define START_TIMER \
	unsigned long long int a,b; \
	a=rdtsc();\
	b=rdtsc();\
	unsigned long long int overhead=b-a;\
	a=rdtsc();

#define STOP_TIMER \
	b=rdtsc();\
	printf("%llu %llu\n",b-a-overhead,overhead);


//windows fixes
#ifdef WIN32

#define timersub(a,b,result) \
  {                                                                        \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;                             \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;                          \
    if ((result)->tv_usec < 0) {                                              \
      --(result)->tv_sec;                                                     \
      (result)->tv_usec += 1000000;                                           \
    }                                                                         \
  }

double drand48() {
    return rand()/(double)RAND_MAX;
}

void srand48(int seed) {
	srand(seed);
}

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS) || defined(__WATCOMC__)
  #define DELTA_EPOCH_IN_USEC  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_USEC  11644473600000000ULL
#endif

#ifndef u_int64_t
	#define u_int64_t unsigned long long
#endif

static u_int64_t filetime_to_unix_epoch (const FILETIME *ft)
{
    u_int64_t res = (u_int64_t) ft->dwHighDateTime << 32;
    res |= ft->dwLowDateTime;
    res /= 10;                   /* from 100 nano-sec periods to usec */
    res -= DELTA_EPOCH_IN_USEC;  /* from Win epoch to Unix epoch */
    return (res);
}

int gettimeofday(struct timeval *tv, void *tz)
{
	FILETIME  ft;
	u_int64_t tim;
	if (!tv) {
	    errno = EINVAL;
	    return (-1);
	}
	GetSystemTimeAsFileTime (&ft);
	tim = filetime_to_unix_epoch (&ft);
	tv->tv_sec  = (long) (tim / 1000000L);
	tv->tv_usec = (long) (tim % 1000000L);
	return (0);
}
#endif  //endif WIN32

typedef struct vector {
	float x,y,z,w;
} vector;

//tangential force on particles
vector __attribute__ ((aligned (16))) tanforce[N];
//particles position
vector __attribute__ ((aligned (16))) pos[N];
//constant vectors
vector __attribute__ ((aligned (16))) Ka_vec={Ka,Ka,Ka,Ka};
vector __attribute__ ((aligned (16))) Kr_vec={Kr,Kr,Kr,Kr};
//generic particle and force
vector *p,*ft;

//scalar product between two vectors
inline float dot_product(vector *v1,vector *v2) {
	float f = 0;
	asm("movaps  (%%eax), %%xmm0\n\t"
		"mulps   (%%ebx), %%xmm0\n\t"
#ifdef USE_SSE3
		"haddps  %%xmm0, %%xmm0\n\t"
		"haddps  %%xmm0, %%xmm0\n\t"
		"movss   %%xmm0, (%%ecx)\n\t"
#else
		"movhlps %%xmm0, %%xmm1\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"shufps  $1, %%xmm0, %%xmm0\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"movss   %%xmm1, (%%ecx)\n\t"
#endif
		::"a"(v1),"b"(v2),"c"(&f):"xmm0","xmm1"
	);
	return f;
}

//squared euclidean norm of a vector
inline float norm2(vector *v) {
	float f;
	asm("movaps (%%eax), %%xmm0\n\t"
		"mulps   %%xmm0, %%xmm0\n\t"
		"movhlps %%xmm0, %%xmm1\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"shufps  $1, %%xmm0, %%xmm0\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"movss   %%xmm1, (%%ecx)\n\t"
		::"a"(v),"c"(&f):"xmm0","xmm1"
	);
	return f;
}

//euclidean norm of a vector
inline float norm(vector *v) {
	float f;
	asm("movaps (%%eax), %%xmm0\n\t"
		"mulps   %%xmm0, %%xmm0\n\t"
		"movhlps %%xmm0, %%xmm1\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"shufps  $1, %%xmm0, %%xmm0\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"sqrtss  %%xmm1, %%xmm1\n\t"
		"movss   %%xmm1, (%%ecx)\n\t"
		::"a"(v),"c"(&f):"xmm0","xmm1"
	);
	return f;
}

//computes sum between two vectors: vector v1=v1+v2
inline void vector_add(vector *v1,vector *v2) {
	asm("movaps (%%eax),%%xmm0\n\t"
		"addps  (%%ebx),%%xmm0\n\t"
		"movaps %%xmm0,(%%eax)\n\t"
		::"a"(v1),"b"(v2):"xmm0","memory"
	);
}

//computes difference between two vectors: vector v1=v1-v2
inline void vector_sub(vector *v1,vector *v2) {
	asm("movaps (%%eax),%%xmm0\n\t"
		"subps  (%%ebx),%%xmm0\n\t"
		"movaps %%xmm0,(%%eax)\n\t"
		::"a"(v1),"b"(v2):"xmm0","memory"
	);
}

//multiplies vector for another vector. v1=v1*v2
inline void vector_mul(vector *v1,vector *v2) {
	asm("movaps (%%eax),%%xmm0\n\t"
		"mulps  (%%ebx),%%xmm0\n\t"
		"movaps %%xmm0,(%%eax)\n\t"
		::"a"(v1),"b"(v2):"xmm0","memory"
	);
}

//multiplies vector for a number
inline void vector_scalar_mul(vector *v1,float f) {
	asm("movaps  (%%eax),%%xmm0\n\t"
		"movss  (%%ebx),%%xmm1\n\t"
		"shufps $0, %%xmm1,%%xmm1\n\t"
		"mulps  %%xmm1, %%xmm0\n\t"
		"movaps %%xmm0,(%%eax)\n\t"
		::"a"(v1),"b"(&f):"xmm0","xmm1","memory"
	);
}

//divides vector for a another vector. v1=v1/v2
inline void vector_div(vector *v1,vector *v2) {
	asm("movaps (%%eax),%%xmm0\n\t"
#ifdef OPTIMIZE_SSE
		"movaps (%%ebx),%%xmm1\n\t"
		"rcpps  %%xmm1, %%xmm1\n\t"
		"mulps  %%xmm1, %%xmm0\n\t"
#else
		"divps  (%%ebx),%%xmm0\n\t"
#endif
		"movaps %%xmm0,(%%eax)\n\t"
		::"a"(v1),"b"(v2):"xmm0","xmm1","memory"
	);
}

//divides vector for a number
inline void vector_scalar_div(vector *v1,float f) {
	asm("movaps  (%%eax),%%xmm0\n\t"
		"movss  (%%ebx),%%xmm1\n\t"
//#ifdef OPTIMIZE_SSE
//		"rcpss  %%xmm1, %%xmm1\n\t"
//		"shufps $0, %%xmm1,%%xmm1\n\t"
//		"mulps  %%xmm1, %%xmm0\n\t"
//#else
		"shufps $0, %%xmm1,%%xmm1\n\t"
		"divps  %%xmm1, %%xmm0\n\t"
//#endif
		"movaps %%xmm0,(%%eax)\n\t"
		::"a"(v1),"b"(&f):"xmm0","xmm1","memory"
	);
}

//computes distance between two vectors
inline float distance(vector *v1,vector *v2) {
	float d;
	asm("movaps (%%eax),%%xmm0\n\t"
		"subps  (%%ebx), %%xmm0\n\t"
		"mulps   %%xmm0, %%xmm0\n\t"
		"movhlps %%xmm0, %%xmm1\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"shufps  $1, %%xmm0, %%xmm0\n\t"
		"addss   %%xmm0, %%xmm1\n\t"
		"sqrtss  %%xmm1, %%xmm1\n\t"
		"movss   %%xmm1, (%%ecx)\n\t"
		::"a"(v1),"b"(v2),"c"(&d):"xmm0","xmm1","memory"
	);
	return d;
}

//normalizes a vector on the surface of the sphere
inline void vector_normalize(vector *v) {
	vector_scalar_div(v,norm(v));
}

//optimized normalization (11 bits precision)
inline void vector_normalize2(vector *v) {
	asm("movaps  (%%eax), %%xmm0\n\t"
		"movaps  %%xmm0,  %%xmm2\n\t"
		"mulps   %%xmm0,  %%xmm0\n\t"
		"movaps  %%xmm0,  %%xmm1\n\t"
		"shufps  $0x4e, %%xmm1, %%xmm0\n\t"
		"addps   %%xmm1, %%xmm0\n\t"
		"movaps  %%xmm0, %%xmm1\n\t"
		"shufps  $0x11, %%xmm1, %%xmm1\n\t"
		"addps   %%xmm1, %%xmm0\n\t"
		"rsqrtps %%xmm0, %%xmm0\n\t"
		"mulps   %%xmm0, %%xmm2\n\t"
		"movaps  %%xmm2, (%%eax)\n\t"
		::"a"(v):"xmm0","xmm1","xmm2","memory");
}

//generates a float number uniformly distributed between min(included) and max(excluded)
float random_float(float min,float max) {
	return drand48()*(max-min)+min;
}

//place a vector randomly on a sphere (uniformly distributed on the surface)
void vector_randomize(vector *p) {
	//z axis coordinate randomization (south-north poles line)
	float z=random_float(-1,1);
	float fi=acos(z);
	//azimuth angle in radians, identifies a semicircumference parallel to z-axis
	float theta=random_float(-M_PI,M_PI);
	p->x=cos(theta)*sin(fi);
	p->y=sin(theta)*sin(fi);
	p->z=z;
}

//calculates angle between two vectors, expressed in degrees
float angle(vector *p,vector *q) {
	static float cf=180/M_PI;
	return cf*acos(dot_product(p,q));
}

//prints vector info
void print_vector(vector *p) {
	printf("{%f,%f,%f,%f} %f\n",p->x,p->y,p->z,p->w,norm(p));
}

//prints a memory dump bit to bit
void bitdump(void *p,unsigned int len) {
	int i,j,k;
	unsigned char *pc=(unsigned char*)p;
	int c=0;
	int row=8;
	for (i=0;i<=len/row && c<len;i++) {
		printf("%04d: ",(i*row));
		for (j=0;j<row/2 && c<len;j++) {
			for (k=0;k<8;k++)
				printf("%d",!!(pc[c] & (0x80>>k)));
			printf(" ");
			c++;
		}
		for (j=0;j<row/2 && c<len;j++) {
			for (k=0;k<8;k++)
				printf("%d",!!(pc[c] & (0x80>>k)));
			printf(" ");
			c++;
		}
		printf("\n");
	}
}

//prints a memory dump byte to byte
void dump(void *p,unsigned int len) {
	unsigned int i,j;
	unsigned char *pc=(unsigned char*)p;
	int c=0;
	int row=16;
	for (i=0;i<len/row && c<len;i++) {
		printf("%04d: ",(i*row));
		for (j=0;j<row/2 && c<len;j++) printf("%02x ",pc[c++]);
		printf(" ");
		for (j=0;j<row/2 && c<len;j++) printf("%02x ",pc[c++]);
		printf("\n");
	}
}

//calculates the potential of actual configuration
//the potential is defined as the sum of inverse of reciprocal distances
float potential() {
	int i,j;
	float ret=0.0;
	for (i=0;i<N;i++)
		for (j=i+1;j<N;j++)
			ret+=1.0/distance(&pos[i],&pos[j]);
	return ret;
}

//potential normalized to N
float single_potential() {
	int i,j;
	float ret=0.0;
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			if (i==j) continue;
			else ret+=1.0/distance(&pos[i],&pos[j]);
	return ret/N;
}

//get current system date
float gettime() {
	static struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec+1e-6*tv.tv_usec;
}

//get tick count directly from processor
extern __inline__ unsigned long long int rdtsc() {
	unsigned long long int x;
	__asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
	return x;
}

//calculates the force vector that v1 applies to v2 (?)
inline void get_force(vector *v1,vector *v2,vector *f) {
	asm(
		/* distance vector computation */
		"movaps (%%eax), %%xmm0\n\t"   //xmm0=v1
		"subps  (%%ebx), %%xmm0\n\t"   //xmm0=v1-v2
		/* distance computation */
		"movaps  %%xmm0, %%xmm1\n\t"   //xmm1=v1-v2
		"mulps   %%xmm1, %%xmm1\n\t"   //xmm1=(v1-v2)*(v1-v2)
		"movhlps %%xmm1, %%xmm2\n\t"
		"addss   %%xmm1, %%xmm2\n\t"
		"shufps  $1, %%xmm1, %%xmm1\n\t"
		"addss   %%xmm1, %%xmm2\n\t"   //xmm2 = distance^2
		/* force vector computation */
#ifdef OPTIMIZE_SSE
		"rsqrtss %%xmm2, %%xmm1\n\t"   // xmm2 =~ distance^-1
		"rcpss   %%xmm2, %%xmm2\n\t"   // xmm1 =~ distance^-2
		"mulss   %%xmm1, %%xmm2\n\t"   // xmm2 =~ distance^-3
		"shufps  $0, %%xmm2, %%xmm2\n\t"
		"mulps   %%xmm2, %%xmm0\n\t"   // xmm0 =~ (v1-v2)·distance^-3 = (v1-v2)/distance^3
#else
		"sqrtss  %%xmm2, %%xmm1\n\t"   // xmm1 = distance
		"mulss   %%xmm1, %%xmm2\n\t"   // xmm2 = distance^3
		"shufps  $0, %%xmm2, %%xmm2\n\t"
		"divps   %%xmm2, %%xmm0\n\t"   // xmm0 = (v1-v2)/distance^3
#endif
		"movaps  %%xmm0, (%%ecx)\n\t"  // assign the result to f
		::"a"(v1),"b"(v2),"c"(f):"xmm0","xmm1","xmm2","memory"
	);
}

//tranforms force vector f in its tangential component vector applied to particle p
inline void make_tangential(vector *f, vector *p) {
	asm("movaps  (%%eax), %%xmm0\n\t"  //xmm0 = f
		"movaps  %%xmm0,  %%xmm2\n\t"  //xmm2 = f
		"movaps  (%%ebx), %%xmm3\n\t"  //xmm3 = p
		"mulps   %%xmm3,  %%xmm0\n\t"  //xmm0 = p
		"movhlps %%xmm0,  %%xmm1\n\t"
		"addss   %%xmm0,  %%xmm1\n\t"
		"shufps  $1, %%xmm0, %%xmm0\n\t"
		"addss   %%xmm0,  %%xmm1\n\t"
		"shufps  $0, %%xmm1, %%xmm1\n\t" //xmm1 = dot product
		"mulps   %%xmm1,  %%xmm3\n\t"
		"subps   %%xmm3,  %%xmm2\n\t"
		"movaps  %%xmm2, (%%eax)\n\t"
		::"a"(f),"b"(p):"xmm0","xmm1","xmm2","xmm3","memory"
	);
}

//very fast square root
inline float fast_sqrt(float f) {
	asm("movss  (%%eax), %%xmm0\n\t"
		"sqrtss  %%xmm0, %%xmm0\n\t"
		"movss   %%xmm0, (%%eax)\n\t"
		::"a"(&f):"xmm0","memory"
	);
	return f;
}

//extremely fast square root (but less precise)
inline float very_fast_sqrt(float f) {
	asm("movss  (%%eax), %%xmm0\n\t"
		"rsqrtss %%xmm0, %%xmm0\n\t"
		"rcpss   %%xmm0, %%xmm0\n\t"
		"movss   %%xmm0, (%%eax)\n\t"
		::"a"(&f):"xmm0","memory"
	);
	return f;
}

int main(int argc,char **argv) {
#ifndef DEBUG
	//random source initialization
	srand48(time(NULL));
#endif
	//particle indices
	int i,j;
	//force between two particles
	vector __attribute__ ((aligned (16))) force;
	//main loop counter
	unsigned int c=CYCLES;
	//random initialization of vectors on the sphere
	for (j=0;j<N;j++)
		vector_randomize(pos+j);
	//cycles we are going to simulate
	int cycles=CYCLES;
	//reset all tangential forces
	memset(tanforce,0,sizeof(tanforce));
	//print the initial state
/*	for (i=0;i<N;i++)
		printf("particle %d: ",i),print_vector(pos+i);*/
	printf("Initial potential: %f\n",potential());
	printf("Starting simulation with %d particles\n",N);
	printf("Maximum number of cycles: %d\n",cycles);
	//time stuff
	struct timeval start,end,time;
	timerclear(&start);
	timerclear(&end);
	gettimeofday(&start,NULL);

	//the main loop
	while (c--) {
#ifdef DEBUG
		printf("\nStep %d\n",CYCLES-c);
#endif
		//compute tangential force on each particle
		for (i=0,p=pos,ft=tanforce;i<N;i++) {
			for (j=i+1;j<N;j++) {
				//compute force vector from particle i on j (?)
				get_force(p,pos+j,&force);     // force = (pos[i]-pos[j]) / |pos[i]-pos[j]|^3
				//add force on particle i
				vector_add(ft,&force);         // tanforce[i] = tanforce[i] + force
				//subtract force on particle j (equal and opposite)
				vector_sub(tanforce+j,&force); // tanforce[j] = tanforce[j] - force
			}
			//repulsive constant scaling
			vector_mul(ft,&Kr_vec);
			//calculation of tangential component
			make_tangential(ft,p);  // ft = ft - (ft·p)*p
			p++; ft++;
		}
		
		//apply calculated forces
		for (i=0;i<N;i++) {
#ifdef DEBUG
			printf("tangential force on particle %d: ",i);
			print_vector(tanforce+i);
			//scalar between tangential force and particle product MUST be 0
			printf("scalar product = %f\n",dot_product(&tanforce[i],&pos[i]));
#endif
			//position adjustment of particle i
			vector_add(pos+i,tanforce+i);
			//re-normalization on the surface of the sphere
// START_TIMER
#ifdef OPTIMIZE_NORMALIZATION
			vector_scalar_div(pos+i,1.0+norm2(tanforce+i)/2.0);
#else
			vector_scalar_div(pos+i,sqrt(1.0+norm2(tanforce+i)));
#endif
// STOP_TIMER
			//save a fraction of force for the next cycle in order to simulate
			//inertia (memory) and friction (for energy loss)
			vector_mul(tanforce+i,&Ka_vec);
		}
#ifdef PRINT_INTERVAL
		if (!(c%PRINT_INTERVAL)) {
			printf("cycle: %d, potential: %0.20f, single: %0.20f\n",(cycles-c),potential(),single_potential());
		}
#endif
#ifdef DEBUG
		for (i=0;i<N;i++) {
			printf("%d: ",i),print_vector(pos+i);
		}
		printf("potential: %0.20f\n",potential());
		printf("Press return to continue...");
		getchar();
#endif
	} //end of main loop

	//print some statistics
	gettimeofday(&end,NULL);
	timersub(&end,&start,&time);
	float t=time.tv_sec+time.tv_usec*1e-6;
	printf("Average cycle time %f us\n",1e6*t/cycles);
	printf("%f cycles/sec\n",cycles/t);
	printf("Final potential %0.20f\n",potential());
	printf("Final configuration:\n");
	for (i=0;i<N;i++) {
		printf("particle %d: ",i);
		print_vector(pos+i);
	}
	exit(0);
}
