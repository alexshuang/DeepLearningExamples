#!/bin/sh

TOP_DIR=../
OUT_DIR=$1
STEPS=${2:-120}
WARMUP_STEPS=${3:-20}
if [ $STEPS -le $WARMUP_STEPS ]; then
    WARMUP_STEPS=$(expr $STEPS / 5)
fi
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

CMD="python3.6 ./multiproc.py --nproc_per_node 1 ./main.py $OUT_DIR --raport-file raport.json -j16 -p 100 --data-backend syntetic --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 --workspace $OUT_DIR -b 128 --amp --static-loss-scale 128 --optimizer-batch-size 1024 --lr 1.024 --mom 0.875 --lr-schedule cosine --epochs  90 --warmup 8 --wd 6.103515625e-05 --memory-format nhwc"

# training
$CMD --warmup-steps $WARMUP_STEPS --max-steps $STEPS --result-dir $OUT_DIR

# record kernels
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}
$CMD --max-steps 3 --warmup-steps 0
rm -f ${OUT_DIR}/*.db ${OUT_DIR}/*.json ${OUT_DIR}/*.txt

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi
unset ROCBLAS_LAYER
NUM_GEMM_PER_ITER=378
if [ -e /data/resnet_rb.csv ]; then
	cp /data/resnet_rb.csv /tmp/rb.csv -v
fi
tail -$NUM_GEMM_PER_ITER $ROCBLAS_LOG_BENCH_PATH > /tmp/rb.csv
sh /tmp/rb.csv | tee /tmp/rb_res.txt
sed -E -n '/(^N,|^T,)/p' /tmp/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."

cp /tmp/rb.csv /data/se-resnext101_rb.csv -v
