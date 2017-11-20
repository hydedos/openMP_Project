// Jacobi.c

#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#define IDX(x, i, j) ((i)*(x)+(j))
int x, y;
int columns;
int rows;
double *grid1;
double *grid2;
double maxdiff = 99999.9;
int numIters = 0;

void *worker(int eps) {

    double temp = 0.0;

    while (maxdiff > eps) {
        for (int i = 1; i < columns; i++) {
          for (int j = 1; j < rows; j++) {
            grid2[IDX(x,i,j)] = (grid1[IDX(x,i-1,j)] + grid1[IDX(x,i+1,j)] +
                     grid1[IDX(x,i,j-1)] + grid1[IDX(x,i,j+1)]) * 0.25;
          }
        }
        numIters++;
        for (int i = 1; i < columns; i++) {
          for (int j = 1; j < rows; j++) {
            grid1[IDX(x,i,j)] = (grid2[IDX(x,i-1,j)] + grid2[IDX(x,i+1,j)] +
                     grid2[IDX(x,i,j-1)] + grid2[IDX(x,i,j+1)]) * 0.25;
          }
        }
        numIters++;

        // maxdiff reduction calculation
        /* compute the maximum difference into global variable */
        maxdiff=0;
        for (int i = 1; i < columns; i++) {
          for (int j = 1; j < rows; j++) {
            temp = grid1[IDX(x,i,j)]-grid2[IDX(x,i,j)];
            if (temp < 0)
              temp = -temp;
            if (maxdiff <  temp)
              maxdiff = temp;
          }
        }
    }
    return NULL;
}
void InitializeGrids(double *grid1, double *startgrid1, double *grid2, double *startgrid2) {
  int i, j;
  double d;
  //printf("Started initializing grids\n");
  for (i = 0; i < y; i++)
    for (j = 0; j < x; j++) {
      if (!scanf("%lg",&d)) {
        perror("scanf");
        exit(1);
      }
      grid1[IDX(x,i,j)] = d;
      startgrid1[IDX(x,i,j)] = d;
      /* put the boundary values into the second grid as well they never change      */
    if ((j == 0) || (j == columns) || (i == 0) || (i == rows)) {
      grid2[IDX(x,i,j)] = grid1[IDX(x,i,j)];
      startgrid2[IDX(x,i,j)] = grid1[IDX(x,i,j)];
    } else {
      grid2[IDX(x,i,j)] = 0.0;
      startgrid2[IDX(x,i,j)] = 0.0;
    } // don't really need to do this but I want clean numbers on
      // debug output
    }
    //printf("Finished initializing grids");
  }

int main(int argc, char* argv[]) {

    if (argc != 5) {
        fprintf(stderr, "Usage:\n%s <threads> <epsilon> <rows> <columns>\n", argv[0]);
        exit(1);
    }

    int threads = atoi(argv[1]);
    double epsilon = atof(argv[2]);
    x = atoi(argv[3]);
    y = atoi(argv[4]);
    rows = x - 1;
    columns = y - 1;

    grid1=(double *) malloc(sizeof(double) * x * y);
    grid2=(double *) malloc(sizeof(double) * x * y);
    double* matrix_1 = malloc(sizeof(double) * x * y);
    double* matrix_2 = malloc(sizeof(double) * x * y);

    InitializeGrids(grid1,matrix_1,grid2,matrix_2);

    //double temp = 0.0;

    // // Fill matrix with values from standard in
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < columns; j++) {
    //         if (!scanf("%lf", &temp)) {
    //             perror("scanf");
    //             exit(1);
    //         }
    //         matrix_1[i * columns + j] = temp;
    //         matrix_2[i * columns + j] = temp;
    //     }
    // }


    struct timeval start;
    struct timeval end;
    struct timeval result;

    gettimeofday(&start, NULL);
    // printf("%d = rows\n", rows);
    // printf("%d = columnss\n", columns);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            printf("%lf ", grid1[IDX(x,i,j)]);
        }
        printf("\n");
    }

    // Start computation here
    worker(epsilon);

    gettimeofday(&end, NULL);
    timersub(&end, &start, &result);


    printf("Converged after %d iterations\n", numIters);
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            printf("%lf ", grid2[IDX(x,i,j)]));
        }
        printf("\n");
    }

    free(matrix_1);
    free(matrix_2);

    fprintf(stderr, "%d, %lf, %d, %2ld, %7ld\n", threads, epsilon, numIters, result.tv_sec, result.tv_usec);
    return 0;
}
