﻿#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

const int N = 1000  //矩阵大小
const int NUM_THREADS = 4  //线程数

double A[N][N + 1];  // 增广矩阵
pthread_barrier_t barrier;  // 定义 barrier 变量

void* eliminate(void* arg) {
	int thread_id = *((int*)arg);
	int start = thread_id * (N / NUM_THREADS);
	int end = (thread_id + 1) * (N / NUM_THREADS);

	for (int k = 0; k < N - 1; k++) {
		for (int i = start + 1; i < end; i++) {
			double factor = A[i][k] / A[k][k];
			for (int j = k; j < N + 1; j++) {
				A[i][j] -= factor * A[k][j];
			}
		}
		// 等待其他线程完成当前列的消元
		pthread_barrier_wait(&barrier);
	}

	return NULL;
}

int main() {
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];
	struct timespec start, end;

	// 初始化 barrier
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);

	// 初始化增广矩阵A
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N + 1; j++) {
			A[i][j] = rand() % 10;  // 随机初始化值
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &start); // 记录开始时间

	// 创建线程
	for (int i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		pthread_create(&threads[i], NULL, eliminate, &thread_ids[i]);
	}

	// 等待所有线程完成
	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	clock_gettime(CLOCK_MONOTONIC, &end); // 记录结束时间

	double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9; // 转换为秒

	printf("Total time: %f seconds\n", time_taken); // 输出总时间

	pthread_barrier_destroy(&barrier);

	return 0;
}