#include <stdio.h>
#include <stdlib.h>

int main() {
    int N = 10000000;
    float *a, *b, *result;
    int i;

  //memory allocation
   a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    result = (float*)malloc(sizeof(float) * N);

    // Initialize matrices
     for(i = 0; i < N; i++) {
          a[i] = i;
          b[i] = i;
     }

    // Add matrices
    for(i = 0; i < N; i++) {
        result[i] = a[i] + b[i];
    }
	
   
     free(a);
     free(b);
     free(result);

     return 0;
}












































