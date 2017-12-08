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
#include <string.h>
#define IDX(x, i, j) ((i)*(x)+(j))
#define COORDINATOR 0
int x, y;
int threads;
double epsilon;
int columns;
int rows;
int numIters = 0;
char *filename;
static void Coordinator(int numWorkers, int x, int y, double epsilon);
void printGrid(double *grid, int x, int y);

struct slices calculateSlice(int worker_id, int num_workers);

/* holds slice information */
struct slices {
  int from;
  int to;
};


void *worker(int id, int numWorkers, int thread_count, double eps) {
  double maxdiff = 99999.9;
  /* get slice information */
  struct slices slice = calculateSlice(id-1, numWorkers);
  int height = slice.to - slice.from + 2;

   // allocate memory for own share of grid
  double *grid1;
  double *grid2;
  grid1 = (double *) malloc(sizeof(double) * x * height);
  grid2 = (double *) malloc(sizeof(double) * x * height);
  /* get grid from cooridnator */
  MPI_Recv(grid1, x * height, MPI_DOUBLE, COORDINATOR, 0, MPI_COMM_WORLD, NULL);
  memcpy(grid2, grid1, x*height*sizeof(double));

  //printGrid(grid1, x, height);

  int fake_bottom_row_index = 0;
  int fake_top_row_index = (height-1) * x;
  int real_bottom_row_index = 1 * x;
  int real_top_row_index = (height-2) * x;

  //   printf("id=%d, bottom index=%d, top_index=%d\n", id, bottom_row_index, top_row_index);
  //printf("x=%d, slice.from=%d, slice.to=%d\n", x, slice.from, slice.to);
  //printf("fakebottom=%d, faketop=%d\n", fake_bottom_row_index, fake_top_row_index);
  //printf("fakebottom=%d, faketop=%d\n", fake_bottom_row_index, fake_top_row_index);




  /* end MPI Test*/

  int i, j;
  double temp = 0.0;
  int notConverged = 1;

  #pragma omp parallel num_threads(thread_count) shared(numIters, grid1, grid2)
  {
    //printf("doing calculations \n");
    while (notConverged) {
      #pragma omp for private (i, j, temp)
        for (i = 1; i < height-1; i++) {
          //printf("row=%d\n", i);
          for (j = 1; j < columns-1; j++) {
           // printf("col=%d\n", j);
            grid2[IDX(x,i,j)] = (grid1[IDX(x,i-1,j)] + grid1[IDX(x,i+1,j)] +
                     grid1[IDX(x,i,j-1)] + grid1[IDX(x,i,j+1)]) * 0.25;
            //printf("%f ", grid2[IDX(x,i,j)]);
            //printf("%f, %f, %f, %f\n", grid1[IDX(x,i-1,j)], grid1[IDX(x,i+1,j)],
           //          grid1[IDX(x,i,j-1)], grid1[IDX(x,i,j+1)]);
          }
        }

    //printf("swapping grids\n");



      /* increase iteration count */
      #pragma omp single
      {
        numIters = numIters + 1;

        /* swap grids */
        double *tempGrid = grid1;
        grid1 = grid2;
        grid2 = tempGrid;
      }

      //printf("exchanging top and bottom \n");
      #pragma omp single
      {
        double *fake_bottom_pointer = grid1 + fake_bottom_row_index;
        double *fake_top_pointer = grid1 + fake_top_row_index;
        double *real_bottom_pointer = grid1 + real_bottom_row_index;
        double *real_top_pointer = grid1 + real_top_row_index;

        //printf("printing real_top_pointer \n");
        //printGrid(real_top_pointer, x, 1);
        //printf("\n");

        //printf("printing real_bottom_pointer \n");
        //printGrid(real_bottom_pointer, x, 1);
        //printf("\n");
        /* send top and bottom rows */
        if (id % 2 == 0) {
          /* recv down */
          if (id > 1) {
            MPI_Recv(fake_bottom_pointer, x, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          /* recv up */
          if (id < numWorkers) {
            MPI_Recv(fake_top_pointer, x, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }

          /* send down */
          if (id > 1) {
            MPI_Send(real_bottom_pointer, x, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD);
          }
          /* send up */
          if (id < numWorkers) {
            MPI_Send(real_top_pointer, x, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
          }
          /* rec down recv up
             send down send up */
        } else {
          /* send up */
          if (id < numWorkers) {
            MPI_Send(real_top_pointer, x, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
          }
          /* send down */
          if (id > 1) {
            MPI_Send(real_bottom_pointer, x, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD);
          }

          /* recv up */
          if (id < numWorkers) {
            MPI_Recv(fake_top_pointer, x, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
          /* recv down */
          if (id > 1) {
            MPI_Recv(fake_bottom_pointer, x, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }
      }

      /* compute the maximum difference into global variable */
      maxdiff=0;
      #pragma omp for reduction(max: maxdiff)
      for (i = 1; i < height-2; i++) {
        for (j = 1; j < columns-1; j++) {
          temp = grid1[IDX(x,i,j)]-grid2[IDX(x,i,j)];
          if (temp < 0)
            temp = -temp;
          if (maxdiff <  temp)
            maxdiff = temp;
        }
      }

      /* communicate using MPI */
      #pragma omp single
      {
        //printGrid(grid1, x, height);
        /* send max diff */
	//printf("%lf \n",maxdiff);
        MPI_Send(&maxdiff, 1, MPI_DOUBLE, COORDINATOR, 0, MPI_COMM_WORLD);
        /* find out if we converged */
        MPI_Bcast(&notConverged, 1, MPI_INT, COORDINATOR, MPI_COMM_WORLD);

      }
    }
    #pragma omp single
    {
      if (id == 1) {
        fprintf(stderr, "%d, %lf, %d", threads, epsilon, numIters);
      }
    }
  }
  /* send grid to coordiinator */
    MPI_Send(grid1+x, x * (height-2), MPI_DOUBLE, COORDINATOR, 0, MPI_COMM_WORLD);
    //printGrid(grid1, x, height);
  return NULL;
}
void InitializeGrids(double *grid1) {
  /* open file */
  char mode = 'r';
  FILE *f = fopen(filename, &mode);

  int i, j;
  double d;
  //printf("Started initializing grids\n");
  for (i = 0; i < y; i++)
    for (j = 0; j < x; j++) {
      if (!fscanf(f, "%lg",&d)) {
        perror("scanf");
        exit(1);
      }
      grid1[IDX(x,i,j)] = d;
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
  numWorkers--;

  if (argc != 6) {
    fprintf(stderr, "Usage:\n%s <threads> <epsilon> <rows> <columns> <file name>\n", argv[0]);
    exit(1);
  }

  threads = atoi(argv[1]);
  epsilon = atof(argv[2]);
  x = atoi(argv[3]);
  y = atoi(argv[4]);
  filename = (argv[5]);
  rows = y;
  columns = x;



  /* read grid from standard in */
  //
  //char procname[MPI_MAX_PROCESSOR_NAME];
  //int length;
  //MPI_Get_processor_name(procname, &length);
  //printf("%s",procname);

  if (myid == COORDINATOR) {
    Coordinator(numWorkers, x, y, epsilon);
    /* do coordinator stuff*/
    /* distribute initial grids */

    /* check max difs */
  } else {
    /* do worker stuff */
    worker(myid, numWorkers, threads, epsilon);
  }

  /* ceanup MPI */
  MPI_Finalize();

  return 0;
}

static void Coordinator(int numWorkers, int x, int y, double epsilon) {
  /* allocate memory for two grids */
  double *grid1;
  grid1=(double *) malloc(sizeof(double) * x * y);

  /* read grid from standard in */
  InitializeGrids(grid1);

  /* timing code for benchmarks */
  struct timeval start;
  struct timeval end;
  struct timeval result;
  gettimeofday(&start, NULL);

   // send grid slices to workers
  for (int i = 1; i <= numWorkers; i++) {
    struct slices slice = calculateSlice(i-1, numWorkers);
    int index = (slice.from-1) * x;
    int count = slice.to - slice.from + 2;
    //printf("attmepting to send to id=%d\n", i);
    MPI_Send(
      grid1+index,
      count * x,
      MPI_DOUBLE,
      i,
      0,
      MPI_COMM_WORLD);
  }

  /* until converged, keep checking maxdifs from workers */
  int notConverged = 1;
  double maxDiff = 0;
  double nodeDiff = 0;
  while (notConverged) {
      for (int i = 1; i <= numWorkers; i++) {
        MPI_Recv(&nodeDiff, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (i == 1) {
          maxDiff = nodeDiff;
        } else {
          if (nodeDiff > maxDiff) {
            maxDiff = nodeDiff;
          }
        }
      }

      //printf("maxdif=%f\n", maxDiff);
      if (maxDiff < epsilon) {
        notConverged = 0;
      }

      /* send convergence info to all nodes */
      MPI_Bcast(
        &notConverged,
        1,
        MPI_INT,
        COORDINATOR,
        MPI_COMM_WORLD);
  }
    //printf("finished loop, gonna get chunks now\n");

    /* end timer */
  gettimeofday(&end, NULL);
  timersub(&end, &start, &result);
  //char sresult[100];
//  snprintf(sresult,100, "%d, %lf, %d", threads, epsilon, numIters);
 // snprintf(sresult,100, "%2ld, %7ld\n", result.tv_sec, result.tv_usec);
//MPI_File f;
  //MPI_File_open(MPI_COMM_WORLD, "data.txt", MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
  //int bufsize = strlen(sresult);
  //MPI_File_write(f, sresult, 100, MPI_DOUBLE, MPI_STATUS_IGNORE);
	//printf("strlen = %d \n", bufsize);
  
  /* collect chunks */
  for (int i = 1; i <= numWorkers; i++) {
    //printf("collecting chunk from id=%d\n", i);
    struct slices slice = calculateSlice(i-1, numWorkers);
    int index = (slice.from) * x;
    int count = slice.to - slice.from;
    MPI_Recv(grid1 + index, count * x, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

 fprintf(stderr,", %2ld, %7ld\n", result.tv_sec, result.tv_usec);
  //printGrid(grid1, x, y);

  free(grid1);
}

void printGrid(double *grid, int x, int y) {
  for (int i = 0; i < y; i++) {
    for (int j = 0; j < x; j++) {
      printf("%lf ", grid[IDX(x,i,j)]);
    }
    printf("\n");
  }
}

/* assumes worker_id starts at zero*/
struct slices calculateSlice(int worker_id, int num_workers) {
  struct slices slice;
  slice.from = worker_id * (rows - 2) / num_workers + 1;
  slice.to = ((worker_id + 1) * (rows - 2)) / num_workers + 1;
  return slice;
}
