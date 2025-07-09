#!/bin/bash
set -euo pipefail

CONDA_PATH="${CONDA_PATH:-$HOME/mambaforge}"
#CONDA_PATH="/home/ubuntu/mambaforge"
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
else
    export PATH="$CONDA_PATH/bin:$PATH"
fi

start_time=$(date +%s)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT" || exit 1

# 参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

input_dir="$1"
output_dir="$2"
model_dir="$PROJECT_ROOT/DNABERT-S"

# 自动创建输入/输出目录
[ ! -d "$input_dir" ] && echo "⚠️ 输入目录不存在，已创建: $input_dir" && mkdir -p "$input_dir"
mkdir -p "$output_dir"

# Step 1: SPAdes 组装
contigs_out="${input_dir}/contigs_over2k.fasta"
if [ ! -f "$contigs_out" ]; then
    conda activate vambnew
    echo "🔧 Step 1: Running SPAdes assembly..."
    python $PROJECT_ROOT/scripts/spades.py -i "$input_dir" -t 66
    echo "✅ SPAdes finished: $contigs_out"
    conda deactivate
else
    echo "⏭️ 已检测到组装结果，跳过 SPAdes: $contigs_out"
fi

# Step 2: 特征路径初始化
fasta_file=$(find "$input_dir" -type f -name "*over2k.fasta" | head -n 1)
bam_file=$(find "$input_dir" -type f -name "*_sorted.bam" | head -n 1)

[ ! -f "$fasta_file" ] && echo "❌ ERROR: 未找到fasta" && exit 1
[ ! -f "$bam_file" ] && echo "❌ ERROR: 未找到bam" && exit 1

BASE_NAME=$(basename "$fasta_file" .fasta)
seq_file="${output_dir}/${BASE_NAME}_over2kseq.txt"
fpf_file="${output_dir}/${BASE_NAME}_fpf.npy"

# Step 3: FPF 特征提取
if [ ! -f "$fpf_file" ]; then
    conda activate dnaberts
    echo "🔧 Step 3: 提取 FPF 特征..."
    python $PROJECT_ROOT/scripts/featureExtract_gpu_2.py -md "$model_dir" -fd "$fasta_file" -sd "$seq_file" -dd "$fpf_file"
    echo "✅ fpf_features output: $fpf_file"
    conda deactivate
else
    echo "⏭️ FPF 已存在，跳过提取"
fi

# Step 4: TNF & RPKM
tnf_file="${output_dir}/${BASE_NAME}_tnf.npz"
rpkm_file="${output_dir}/rpkm.npz"
if [ ! -f "$tnf_file" ] || [ ! -f "$rpkm_file" ]; then
    conda activate vambnew
    echo "🔧 Step 4: 计算 TNF & RPKM..."
    python $PROJECT_ROOT/myvae/mainfiles/calc_tnf_and_rpkm_2.py -od "$output_dir" -fd "$fasta_file" -bam "$bam_file"
    conda deactivate
else
    echo "⏭️ TNF/RPKM 已存在，跳过"
fi

# Step 5: VAE 特征融合
vaef_file="${output_dir}/${BASE_NAME}_vae_features.npy"
if [ ! -f "$vaef_file" ]; then
    conda activate vambnew
    echo "🔧 Step 5: VAE 特征融合..."
    python $PROJECT_ROOT/myvae/mainfiles/vaeTest_2.py -dd "$fpf_file" -td "$tnf_file" -rd "$rpkm_file" -vd "$vaef_file"
    conda deactivate
else
    echo "⏭️ VAE 特征已存在，跳过"
fi

# Step 6: 计算初始聚类数
kmer_file="${output_dir}/4mer.csv"
seqid_file="${output_dir}/seqid.csv"
if [ ! -f "$kmer_file" ] || [ ! -f "$seqid_file" ]; then
    conda activate vambnew
    python $PROJECT_ROOT/scripts/calculate_kmer_multi_thread_2.py "$fasta_file" "$kmer_file" "$seqid_file" 4
    conda deactivate
fi

# Step 7: marker gene 推断聚类数
cvf_file="${output_dir}/cluster_value"
if [ ! -f "$cvf_file" ]; then
    conda activate copygen
    python $PROJECT_ROOT/marker_gene/src/marker_gene_utils.py -kf "$kmer_file" -cf "$fasta_file" -cvf "$cvf_file"
    conda deactivate
fi

# Step 8: 聚类
LABEL_FILE="${output_dir}/${BASE_NAME}_prinum.txt"
BINS_OUTPUT_DIR="${output_dir}/${BASE_NAME}_bins"
mkdir -p "$BINS_OUTPUT_DIR"
if [ ! -f "$LABEL_FILE" ]; then
    conda activate dnaberts
    echo "🔧 Step 8: 聚类..."
    python $PROJECT_ROOT/scripts/myCluster_2.py -vd "$vaef_file" -ld "$LABEL_FILE" -fd "$fasta_file" -bd "$BINS_OUTPUT_DIR" -cvf "$cvf_file"
    conda deactivate
else
    echo "⏭️ 聚类已完成，跳过"
fi

# Step 9: CheckM2 评估
checkm_out="${output_dir}/${BASE_NAME}_checkm2"
if [ ! -f "$checkm_out/quality_report.tsv" ]; then
    conda activate checkm2
    echo "🔧 Step 9: CheckM2 评估..."
    checkm2 predict -t 64 -x fasta --input "$BINS_OUTPUT_DIR" --output-directory "$checkm_out"
    conda deactivate
else
    echo "⏭️ CheckM2 已完成，跳过"
fi

# Step 10: 统计结果
tsv_filea="${checkm_out}/quality_report.tsv"
if [ -f "$tsv_filea" ]; then
    echo "📊 Step 10: 汇总 Bin 质量指标："
    awk -F '\t' 'NR > 1 && $2 >= 90 && $3 < 5 {c1++}
                 NR > 1 && $2 >= 80 && $3 < 5 {c2++}
                 NR > 1 && $2 >= 70 && $3 < 5 {c3++}
                 NR > 1 && $2 >= 60 && $3 < 5 {c4++}
                 NR > 1 && $2 >= 50 && $3 < 5 {c5++}
                 NR > 1 && $2 >  90 && $3 < 5 {c6++}
                 NR > 1 && $2 >= 50 && $3 < 10 {c7++}
                 END {
                    print ">=90% Completeness, <5% Contamination:", c1+0
                    print ">=80% Completeness, <5% Contamination:", c2+0
                    print ">=70% Completeness, <5% Contamination:", c3+0
                    print ">=60% Completeness, <5% Contamination:", c4+0
                    print ">=50% Completeness, <5% Contamination:", c5+0
                    print ">90% Completeness, <5% Contamination:", c6+0
                    print ">=50% Completeness, <10% Contamination:", c7+0
                 }' "$tsv_filea"
fi

# 结束
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "✅ 所有流程结束，总运行时间: $runtime 秒"
