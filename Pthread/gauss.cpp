#include <iostream>
#include <Windows.h>
#include <sys/utime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
//#include <arm_neon.h>
#include <immintrin.h>
#include <pthread.h>
using namespace std;
#define TIME_UTC 1
int N = 300;
int numsOf_threads = 4;
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
float **m;
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
void* LU_thread_row(void* param)
{
    int thread_id = *(int*)param;

    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1 + thread_id; i < N; i += numsOf_threads)
        {
            m[i][k] = m[i][k] / m[k][k];

            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
        }

        pthread_barrier_wait(&barrier_Divsion); // 等待其他线程完成当前行的计算
    }

    pthread_exit(NULL);
}
void LU_row(float** m)
{
    pthread_t threads[numsOf_threads];
    int thread_ids[numsOf_threads];
    pthread_barrier_init(&barrier_Divsion, NULL, numsOf_threads);

    for (int i = 0; i < numsOf_threads; i++)
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, LU_thread_row, (void*)&thread_ids[i]);
    }

    for (int i = 0; i < numsOf_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion);
}

void* LU_thread(void* param)
{
    int thread_id = *(int*)param;

    for (int k = 0; k < N; k++)
    {
        if(thread_id==0)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
        }


        pthread_barrier_wait(&barrier_Divsion); // 等待其他线程完成当前列的计算

        for (int i = k+1+thread_id; i < N; i+=numsOf_threads)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }

        pthread_barrier_wait(&barrier_Elimination); // 等待其他线程完成当前列的计算
    }

    pthread_exit(NULL);
}

void LU(float** m)
{
    pthread_t threads[numsOf_threads];
    int thread_ids[numsOf_threads];
    pthread_barrier_init(&barrier_Divsion, NULL, numsOf_threads);
    pthread_barrier_init(&barrier_Elimination, NULL, numsOf_threads);

    for (int i = 0; i < numsOf_threads; i++)
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, LU_thread, (void*)&thread_ids[i]);
    }

    for (int i = 0; i < numsOf_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrierattr_destroy(&barrier_Elimination);
}
void* LU_AVX_thread(void* param)
{
    int thread_id = *(int*)param;

    for (int k = 0; k < N; k++)
    {
        __m128 vt = _mm_set1_ps(m[k][k]);
        for (int j = k + 1 + thread_id; j < N; j += numsOf_threads)
        {
            if (j + 4 > N)
            {
                for (; j < N; j++)
                {
                    m[k][j] = m[k][j] / m[k][k];
                }
            }
            else
            {
                __m128 va = _mm_loadu_ps(m[k] + j);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(m[k] + j, va);
            }
            m[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Divsion); // Wait for other threads to complete current column calculation

        for (int i = k + 1 + thread_id; i < N; i += numsOf_threads)
        {
            for (int j = k; j < N; j += 4)
            {
                if (j + 4 > N)
                {
                    for (; j < N; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                }
                else
                {
                    __m128 temp1 = _mm_loadu_ps(m[i] + j);
                    __m128 temp2 = _mm_loadu_ps(m[k] + j);
                    __m128 temp3 = _mm_set1_ps(m[i][k]);
                    temp2 = _mm_mul_ps(temp3, temp2);
                    temp1 = _mm_sub_ps(temp1, temp2);
                    _mm_storeu_ps(m[i] + j, temp1);
                }
                m[i][k] = 0;
            }
        }

        pthread_barrier_wait(&barrier_Elimination); // Wait for other threads to complete current column calculation
    }

    pthread_exit(NULL);
}

void LU_AVX(float** m)
{
    pthread_t threads[numsOf_threads];
    int thread_ids[numsOf_threads];
    pthread_barrier_init(&barrier_Divsion, NULL, numsOf_threads);
    pthread_barrier_init(&barrier_Elimination, NULL, numsOf_threads);

    for (int i = 0; i < numsOf_threads; i++)
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, LU_AVX_thread, (void*)&thread_ids[i]);
    }

    for (int i = 0; i < numsOf_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrierattr_destroy(&barrier_Elimination);
}
void *LU_SSE_thread(void *param)
{
    int thread_id = *(int *)param;

    for (int k = 0; k < N; k++)
    {
        if (thread_id == 0)
        {
            for (int j = k + 1; j < N; j++)
            {
                m[k][j] = m[k][j] / m[k][k];
            }
            m[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_Divsion); // Wait for other threads to finish division

        for (int i = k + 1 + thread_id; i < N; i += numsOf_threads)
        {
            for (int j = k; j < N; j += 4)
            {
                if (j + 4 > N)
                {
                    for (; j < N; j++)
                    {
                        m[i][j] = m[i][j] - m[i][k] * m[k][j];
                    }
                }
                else
                {
                    __m128 temp1 = _mm_loadu_ps(m[i] + j);
                    __m128 temp2 = _mm_loadu_ps(m[k] + j);
                    __m128 temp3 = _mm_set1_ps(m[i][k]);
                    temp2 = _mm_mul_ps(temp3, temp2);
                    temp1 = _mm_sub_ps(temp1, temp2);
                    _mm_storeu_ps(m[i] + j, temp1);
                }
                m[i][k] = 0;
            }
        }

        pthread_barrier_wait(&barrier_Elimination); // Wait for other threads to finish elimination
    }

    pthread_exit(NULL);
}
void LU_SSE(float **m)
{
    pthread_t threads[numsOf_threads];
    int thread_ids[numsOf_threads];
    pthread_barrier_init(&barrier_Divsion, NULL, numsOf_threads);
    pthread_barrier_init(&barrier_Elimination, NULL, numsOf_threads);

    for (int i = 0; i < numsOf_threads; i++)
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, LU_SSE_thread, (void *)&thread_ids[i]);
    }

    for (int i = 0; i < numsOf_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrierattr_destroy(&barrier_Elimination);
}
int main()
{

        m = new float *[N];
        m_reset(m);

        LU_row(m);
    return 0;

}
