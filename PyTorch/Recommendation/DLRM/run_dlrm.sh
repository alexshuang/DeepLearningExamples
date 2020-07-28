#!/bin/sh

OUT_DIR=${1:-out}

mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

WARMUP_STEPS=${3:-20}
TRAIN_STEPS=${2:-100}
MAX_STEPS=$(expr $WARMUP_STEPS + $TRAIN_STEPS)

dataset_dir=/data/small_terabyte_dataset

#CMD="python3.6 -m dlrm.scripts.main --mode train --synthetic_dataset true --fp16 true --print_freq 1"
CMD="python3.6 -m dlrm.scripts.main \
  --mode train \
  --seed 8 \
  --epochs 1 \
  --print_freq 1 \
  --batch_size 32768 \
  --bottom_mlp_sizes 512,256,128 \
  --top_mlp_sizes 1024,1024,512,256,1 \
  --embedding_dim 128 \
  --num_numerical_features 13 \
  --dataset $dataset_dir \
  --dataset_type binary \
  --amp \
  --optimized_mlp"

#  --benchmark_warmup_steps 250 \
#  --dataset $dataset_dir \
#  --dataset_type binary \
#  --max_steps 500 \

set -e

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

# split one iteration
NUM_GEMM=26
tail -$NUM_GEMM $ROCBLAS_LOG_BENCH_PATH > /tmp/rb_tail.csv
cp /tmp/rb_tail.csv $ROCBLAS_LOG_BENCH_PATH
sed -n '/Cijk_A/p' ${OUT_DIR}/kernel_prof.csv > /tmp/kname.csv
tail -$NUM_GEMM /tmp/kname.csv > ${OUT_DIR}/kernel_name.csv

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi

unset ROCBLAS_LAYER
sh /tmp/rb_tail.csv 2>&1 > /tmp/rb_res.txt
sed -E -n '/(^N,|^T,)/p' /tmp/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."
