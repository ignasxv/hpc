#include <stdio.h>
#include <math.h>
int main ()
{

int lower, upper,total,i,j;
int isprime;

total = 0;
lower = 2;
upper = 131072;

for (i = lower; i<= upper; i++)
{
  isprime = 1;

     for ( j=2; j<i; j++)
       {
        if (i % j == 0)
          {
           isprime = 0;
           break;
          }
       }
  total+=isprime;     
}

   printf("Number of Prime Numbers from %d to %d=%d\n",lower,upper,total);

}
