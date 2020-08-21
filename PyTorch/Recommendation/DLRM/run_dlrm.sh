#!/bin/bash

# arg 1: gpu device id
# arg 2: output dir
function get_hw_info() {
    output=$2/hw.csv
    clks=(`/opt/rocm/bin/rocm-smi -d $1 --showclkfrq | grep '*' | grep 'GPU' | awk '{print $4}'`)
    dev=`/opt/rocm/bin/rocm-smi -d $1 -i | grep 'GPU' | awk '{print $5}'`
    echo "device,sclk,mclk" > $output
    echo "$dev,${clks[2]},${clks[1]}" >> $output
}

OUT_DIR=${1:-out}
TMP_DIR=$OUT_DIR/tmp

mkdir -p $OUT_DIR $TMP_DIR
#rm -rf $OUT_DIR/*

MAX_STEPS=${2:-1000}
WARMUP_STEPS=${3:-30}
DEVICE=${4:-1}

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

set -e

HIP_VISIBLE_DEVICE=$DEVICE

get_hw_info $DEVICE $OUT_DIR

# end2end perf
$CMD --benchmark_warmup_steps ${WARMUP_STEPS} --max_steps ${MAX_STEPS} | tee $TMP_DIR/run.log
sed -n '/^Epoch:\[0/p' $TMP_DIR/run.log > ${OUT_DIR}/run_res.csv

# record kernels
export ROCBLAS_LAYER=6
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
export ROCBLAS_LOG_PROFILE_PATH=${OUT_DIR}/rocblas_config.json
rm -f ${ROCBLAS_LOG_BENCH_PATH}
rm -f ${ROCBLAS_LOG_PROFILE_PATH}
echo "pmc: FetchSize L2CacheHit" > input.txt
/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${TMP_DIR}/kernel_prof.csv \
$CMD --benchmark_warmup_steps 0 --max_steps 100

# split one iteration
NUM_GEMM=23
tail -$NUM_GEMM $ROCBLAS_LOG_BENCH_PATH > $TMP_DIR/rb_tail.csv
cp $TMP_DIR/rb_tail.csv $ROCBLAS_LOG_BENCH_PATH
sed -n '/Cijk_A/p' ${TMP_DIR}/kernel_prof.csv > $TMP_DIR/gemm_kernel_prof.csv
tail -$NUM_GEMM $TMP_DIR/gemm_kernel_prof.csv > ${OUT_DIR}/kernel_prof.csv

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi
unset ROCBLAS_LAYER
sed "s/$/ -i ${MAX_STEPS} -j ${WARMUP_STEPS}/g" $ROCBLAS_LOG_BENCH_PATH > ${TMP_DIR}/rb.csv
sh ${TMP_DIR}/rb.csv 2>&1 > $TMP_DIR/rb_res.txt | tee $TMP_DIR/rocblas_bench.log
sed -E -n '/(^N,|^T,)/p' $TMP_DIR/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."
