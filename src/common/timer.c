#include "timer.h"

#include <stdio.h>

void timer_start(Timer *t)
{
    if (clock_gettime(CLOCK_MONOTONIC, t) != 0) {
        perror("clock_gettime");
    }
}

double timer_stop_ms(const Timer *t)
{
    struct timespec t2;
    double sec;
    double nsec;

    if (clock_gettime(CLOCK_MONOTONIC, &t2) != 0) {
        perror("clock_gettime");
        return 0.0;
    }

    sec = (double)(t2.tv_sec - t->tv_sec) * 1e3;
    nsec = (double)(t2.tv_nsec - t->tv_nsec) / 1e6;
    return sec + nsec;
}
