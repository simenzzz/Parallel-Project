#ifndef TIMER_H
#define TIMER_H

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct timespec Timer;

void timer_start(Timer *t);
double timer_stop_ms(const Timer *t);

#ifdef __cplusplus
}
#endif

#endif
