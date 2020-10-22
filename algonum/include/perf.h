#ifndef PERF_H
#define PERF_H
#include <sys/time.h>

typedef struct timeval perf_t;

/**
 * @brief Return the perf_t structure at an instant t.
 *
 * Note that the MPI version integrates a Barrier to make sure all processes are
 * synchronized.
 *
 * @param[in,out] p
 *      The allocated perf_t structure to fill-in with the actual time.
 */
void perf( perf_t *p );

/**
 * @brief Return the difference of two perf_t structure.
 *
 * @param[in] begin
 *      The perf_t structure at which the timer has been started.
 *
 * @param[in,out] end
 *      On entry, the perf_t structure at which the timer has been ended.
 *      On exit, contains the difference (end - begin).
 */
void perf_diff( const perf_t *begin, perf_t *end );

/**
 * @brief Print the decomposed time as "m:s:ms:us"
 *
 * @param[in] p
 *      The input perf_t structure with the timer to print.
 */
void perf_printh( const perf_t *p );

/**
 * @brief Print the time in us
 *
 * @param[in] p
 *      The input perf_t structure with the timer to print.
 */
void perf_printmicro( const perf_t *p );

/**
 * @brief Returns the number of MFlop/s
 *
 * @param[in] p
 *      The input perf_t structure with the timer of the operation.
 *
 * @param[in] nb_op
 *      The total number of operations performed during the elapsed time of the
 *      timer p.
 *
 * @return The number of operations per second in MFlop/s.
 *
 */
double perf_mflops(const perf_t *p, const double nb_op);

/**
 * @brief Returns the number of GFlop/s
 *
 * @param[in] p
 *      The input perf_t structure with the timer of the operation.
 *
 * @param[in] nb_op
 *      The total number of operations performed during the elapsed time of the
 *      timer p.
 *
 * @return The number of operations per second in GFlop/s
 *
 */
double perf_gflops(const perf_t *p, const double nb_op);

#endif
