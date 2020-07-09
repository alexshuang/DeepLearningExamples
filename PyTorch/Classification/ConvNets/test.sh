#!/bin/sh

echo "rocprof:"
sed -n '/Cijk_A/p' $1/kernel_prof.csv | wc -l

echo "MI number:"
sed -n '/Cijk_.*_MI/p' $1/kernel_prof.csv | wc -l

echo "rocblas bench:"
cat $1/rocblas_bench.csv | wc -l
