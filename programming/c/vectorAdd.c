#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000

int main(){
    float *a, *b, *out;
   int i,j; 
   // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    
    for ( i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
//vector addition
    for ( j = 0; j < N; j++){
        out[j] = a[j] + b[j];
  }  

 // Test
    for (i = 0; i < 10; i++) {
    printf("The first  10 elements of array out are %f\n", out[i]);
   }

  // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
return 0;
}

