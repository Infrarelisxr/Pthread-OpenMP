#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;
int NUM_THREADS = 8;
int N = 200;
#define parallel TRUE
int i,j,k;
void m_reset(float **m)
{
    for (int i = 0; i < N; i++)
    {
        m[i] = new float[N];
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
        {
            m[i][j] = rand();
        }
        for (int k = 0; k < N; k++)
        {
            for (int i = k + 1; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    m[i][j] += m[k][j];
                }
            }
        }
    }
}
void LU(float** m)
{

    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}
void LU_omp_row(float** m)
{
    float tmp = 0;
    #pragma omp parallel if(parallel), num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            tmp = m[k][k];
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] = m[k][j] / tmp;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for
        for (int i = k + 1; i < N; i++)
        {
            tmp = m[i][k];
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - tmp * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}
void LU_omp_row_dy(float** m)
{
    float tmp = 0;
    #pragma omp parallel if(parallel), num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            tmp = m[k][k];
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] = m[k][j] / tmp;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for schedule(dynamic,50)
        for (int i = k + 1; i < N; i++)
        {
            tmp = m[i][k];
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - tmp * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}
void LU_omp_simd(float** m)
{
    float tmp = 0;
    #pragma omp parallel if(parallel), num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            tmp = m[k][k];
            #pragma omp simd aligned(m : 16) simdlen(4)
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] = m[k][j] / tmp;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for schedule(simd \
                         : guided)
        for (int i = k + 1; i < N; i++)
        {
            tmp = m[i][k];
            #pragma omp simd aligned(m : 16) simdlen(4)
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - tmp * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}
void LU_omp_col(float** m)
{
    #pragma omp parallel num_threads(NUM_THREADS), default(none), private(i, j, k), shared(m, N)
    for(int k = 0;k < N;k++)
    {
        #pragma omp for schedule(simd \
                         : guided)
        for(int j = k+1;j < N;j++)
        {
            m[k][j] = m[k][j]/m[k][k];
            for(int i = k+1;i < N;i++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }
        #pragma omp single
        {
            m[k][k]=1.0;
            for(int i = k + 1;i < N;i++)
            {
                m[i][k] = 0;
            }
        }
    }

}

int main()
{

    float** m = new float* [N];
    m_reset(m);
    struct timespec sts, ets;
    clock_gettime(CLOCK_MONOTONIC, &sts);
    for (int cycle = 0; cycle < 100; cycle++)
    {
        LU_omp_row_dy(m);
    }
    clock_gettime(CLOCK_MONOTONIC, &ets);
    time_t dsec = ets.tv_sec - sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if (dnsec < 0)
    {
        dsec--;
        dnsec+=1000000000ll;
    }
    printf("%lld.%09lld\n",dsec,dnsec);




    return 0;
}
