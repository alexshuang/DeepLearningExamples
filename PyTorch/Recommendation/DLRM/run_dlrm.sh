#!/bin/sh

OUT_DIR=/data/rocm${ROCM_VERSION}_rocblas${ROCBLAS_VERSION}/dlrm

mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

set -e

WARMUP_STEPS=${2:-20}
TRAIN_STEPS=${1:-100}
MAX_STEPS=$(expr $WARMUP_STEPS + $TRAIN_STEPS)

CMD="python3.6 -m dlrm.scripts.main --mode train --synthetic_dataset true --print_freq 1"

# end2end perf
$CMD --benchmark_warmup_steps ${WARMUP_STEPS} --max_steps ${MAX_STEPS} | tee /tmp/run.log
sed -n '/^Epoch:\[0/p' /tmp/run.log > ${OUT_DIR}/run_res.csv

# record kernels
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}

echo "pmc: FetchSize L2CacheHit" > input.txt
/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${OUT_DIR}/kernel_prof.csv \
$CMD --benchmark_warmup_steps 0 --max_steps 1
rm -f ${OUT_DIR}/*.db ${OUT_DIR}/*.json ${OUT_DIR}/*.txt

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi

unset ROCBLAS_LAYER
NUM_GEMM=26
tail -$NUM_GEMM $ROCBLAS_LOG_BENCH_PATH > /tmp/rb_tail.csv
sh /tmp/rb_tail.csv | tee /tmp/rb_res.txt
sed -E -n '/(^N,|^T,)/p' /tmp/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."
