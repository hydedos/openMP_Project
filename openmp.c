// Jacobi.c

#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#define IDX(x, i, j) ((i)*(x)+(j))
#define COORDINATOR 0
int x, y;
int columns;
int rows;
double *grid1;
double *grid2;
double maxdiff = 99999.9;
int numIters = 0;

static void Coordinator(int numWorkers, int x, int y);


void *worker(int thread_count, double eps) {
  /* MPI test */
  int test;
  MPI_Status status;
  MPI_Recv(&test, 1, MPI_INT, COORDINATOR, 0, MPI_COMM_WORLD, &status);
  printf("got value: %d", test);
  return NULL;
  /* end MPI Test*/
  int i, j;
  double temp = 0.0;

  #pragma omp parallel num_threads(thread_count) shared(numIters, grid1, grid2)
  {
    while (maxdiff > eps) {
    #pragma omp for private (i, j, temp)
      for (i = 1; i < columns; i++) {
        for (j = 1; j < rows; j++) {
          grid2[IDX(x,i,j)] = (grid1[IDX(x,i-1,j)] + grid1[IDX(x,i+1,j)] +
                   grid1[IDX(x,i,j-1)] + grid1[IDX(x,i,j+1)]) * 0.25;
        }
      }

      #pragma omp for private (i, j, temp)
      for (i = 1; i < columns; i++) {
        for (j = 1; j < rows; j++) {
          grid1[IDX(x,i,j)] = (grid2[IDX(x,i-1,j)] + grid2[IDX(x,i+1,j)] +
                   grid2[IDX(x,i,j-1)] + grid2[IDX(x,i,j+1)]) * 0.25;
        }
      }
      //
      #pragma omp single
      {
        numIters = numIters + 2;
      }
      // maxdiff reduction calculation
      /* compute the maximum difference into global variable */
      maxdiff=0;
      #pragma omp for reduction(max: maxdiff)
      for (i = 1; i < columns; i++) {
        for (j = 1; j < rows; j++) {
          temp = grid1[IDX(x,i,j)]-grid2[IDX(x,i,j)];
          if (temp < 0)
            temp = -temp;
          if (maxdiff <  temp)
            maxdiff = temp;
        }
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
  /* mpi variables */
  int myid;
  int numWorkers;
  /* setup MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);  /* what is my id (rank)? */
  MPI_Comm_size(MPI_COMM_WORLD, &numWorkers);  /* how many processes? */
  numWorkers--;   /* one coordinator, the other processes are workers */

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

  /* allocate memory for two grids */
  grid1=(double *) malloc(sizeof(double) * x * y);
  grid2=(double *) malloc(sizeof(double) * x * y);
  double* matrix_1 = malloc(sizeof(double) * x * y);
  double* matrix_2 = malloc(sizeof(double) * x * y);

  /* read grid from standard in */
  // InitializeGrids(grid1,matrix_1,grid2,matrix_2);

  /* timing code for benchmarks */
  struct timeval start;
  struct timeval end;
  struct timeval result;
  gettimeofday(&start, NULL);

  /* print the grid we read in */
  // for (int i = 0; i < x; i++) {
  //   for (int j = 0; j < y; j++) {
  //     printf("%lf ", grid1[IDX(x,i,j)]);
  //   }
  //   printf("\n");
  // }

  if (myid == COORDINATOR) {
    Coordinator(numWorkers, x, y);
    /* do coordinator stuff*/
    /* distribute initial grids */

    /* check max difs */
  } else {
    /* do worker stuff */
    worker(threads, epsilon);
  }

  /* ceanup MPI */
  MPI_Finalize();

  gettimeofday(&end, NULL);
  timersub(&end, &start, &result);

  /* print final grid */
  // printf("Converged after %d iterations\n", numIters);
  // for (int i = 0; i < x; i++) {
  //   for (int j = 0; j < y; j++) {
  //     printf("%lf ", grid1[IDX(x,i,j)]);
  //   }
  //   printf("\n");
  // }

  free(matrix_1);
  free(matrix_2);

  fprintf(stderr, "%d, %lf, %d, %2ld, %7ld\n", threads, epsilon, numIters, result.tv_sec, result.tv_usec);
  return 0;
}

static void Coordinator(int numWorkers, int x, int y) {
  int test = 10;
  for (int i = 1; i <= numWorkers; i++) {
    MPI_Send(&test, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    test++;
  }
  /* distribute initial work */

  /*
  w_data[i].start_idx = i * (rows - 2) / threads + 1;
  w_data[i].end_idx = ((i + 1) * (rows - 2)) / threads + 1;
  */

  /* until converged, keep checking maxdifs from workers */
}
