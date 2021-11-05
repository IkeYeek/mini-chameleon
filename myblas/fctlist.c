/**
 *
 * @file fctlist.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Functions to test the dgemm_tiled variants.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "algonum.h"
#include <assert.h>
#include <strings.h>
#include <stdio.h>

fct_list_t *list_ddot   = NULL;
fct_list_t *list_dgemm  = NULL;
fct_list_t *list_dgetrf = NULL;

void
register_fct( fct_list_t *fct, int algo )
{
    assert( fct->next == NULL );
    if ( algo == ALGO_DDOT ) {
        fct->next = list_ddot;
        list_ddot = fct;
    }
    else if ( algo == ALGO_GETRF ) {
        fct->next = list_dgetrf;
        list_dgetrf = fct;
    }
    else {
        fct->next = list_dgemm;
        list_dgemm = fct;
    }
}

fct_list_t *
search_fct( const char *name, int algo )
{
    fct_list_t *item;

    if ( algo == ALGO_DDOT ) {
        item = list_ddot;
    }
    else if (algo == ALGO_GETRF){
        item = list_dgetrf;
    }
    else {
        item = list_dgemm;
    }

    while( item != NULL ) {
        if ( strcasecmp( name, item->name ) == 0 ) {
            return item;
        }
        item = item->next;
    }

    return item;
}

void
print_fct( int algo )
{
    fct_list_t *item;

    if ( algo == ALGO_DDOT ) {
        item = list_ddot;
    }
    else if (algo == ALGO_GETRF){
        item = list_dgetrf;
    }
    else {
        item = list_dgemm;
    }

    while( item != NULL ) {
        printf( "    %-8s %s\n", item->name, item->helper );
        item = item->next;
    }
}
