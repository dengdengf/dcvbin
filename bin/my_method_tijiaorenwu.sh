#!/bin/bash

#SBATCH --job-name=dnaberts
#SBATCH --output=dnaberts_output_test.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4

set -euo pipefail
export LD_LIBRARY_PATH=""

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
    echo "🔧 Step 1: Running SPAdes assembly..."
    conda run -n vambnew -- python $PROJECT_ROOT/scripts/spades.py -i "$input_dir" -t 66
    echo "✅ SPAdes finished: $contigs_out"
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
    echo "🔧 Step 3: 提取 FPF 特征..."
    conda run -n dnaberts -- python $PROJECT_ROOT/scripts/featureExtract_gpu_2.py -md "$model_dir" -fd "$fasta_file" -sd "$seq_file" -dd "$fpf_file"
    echo "✅ fpf_features output: $fpf_file"
else
    echo "⏭️ FPF 已存在，跳过提取"
fi

# Step 4: TNF & RPKM
tnf_file="${output_dir}/${BASE_NAME}_tnf.npz"
rpkm_file="${output_dir}/rpkm.npz"
if [ ! -f "$tnf_file" ] || [ ! -f "$rpkm_file" ]; then
    echo "🔧 Step 4: 计算 TNF & RPKM..."
    conda run -n vambnew -- python $PROJECT_ROOT/myvae/mainfiles/calc_tnf_and_rpkm_2.py -od "$output_dir" -fd "$fasta_file" -bam "$bam_file"
else
    echo "⏭️ TNF/RPKM 已存在，跳过"
fi

# Step 5: VAE 特征融合
vaef_file="${output_dir}/${BASE_NAME}_vae_features.npy"
if [ ! -f "$vaef_file" ]; then
    echo "🔧 Step 5: VAE 特征融合..."
    #conda run -n vambnew python $PROJECT_ROOT/myvae/mainfiles/vaeTest_2.py -dd "$fpf_file" -td "$tnf_file" -rd "$rpkm_file" -vd "$vaef_file"
    #conda run -n vambnew python "$PROJECT_ROOT/myvae/mainfiles/vaeTest_2.py" \-dd "$fpf_file" -td "$tnf_file" -rd "$rpkm_file" -vd "$vaef_file"
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate vambnew
    python "$PROJECT_ROOT/myvae/mainfiles/vaeTest_2.py" \
        -dd "$fpf_file" -td "$tnf_file" -rd "$rpkm_file" -vd "$vaef_file"
else
    echo "⏭️ VAE 特征已存在，跳过"
fi

# Step 6: 计算初始聚类数
kmer_file="${output_dir}/4mer.csv"
seqid_file="${output_dir}/seqid.csv"
if [ ! -f "$kmer_file" ] || [ ! -f "$seqid_file" ]; then
    conda run -n vambnew python $PROJECT_ROOT/scripts/calculate_kmer_multi_thread_2.py "$fasta_file" "$kmer_file" "$seqid_file" 4
fi

# Step 7: marker gene 推断聚类数
cvf_file="${output_dir}/cluster_value"
if [ ! -f "$cvf_file" ]; then
    conda run -n copygen  python $PROJECT_ROOT/marker_gene/src/marker_gene_utils.py -kf "$kmer_file" -cf "$fasta_file" -cvf "$cvf_file"
fi

# Step 8: 聚类
LABEL_FILE="${output_dir}/${BASE_NAME}_prinum.txt"
BINS_OUTPUT_DIR="${output_dir}/${BASE_NAME}_bins"
mkdir -p "$BINS_OUTPUT_DIR"
if [ ! -f "$LABEL_FILE" ]; then
    echo "🔧 Step 8: 聚类..."
    echo "[DEBUG] vaef_file=$vaef_file"
    echo "[DEBUG] label_file=$LABEL_FILE"
    #conda run -n dnaberts -- python $PROJECT_ROOT/scripts/myCluster_2.py -vd "$vaef_file" -ld "$LABEL_FILE" -fd "$fasta_file" -bd "$BINS_OUTPUT_DIR" -cvf "$cvf_file"
    # 激活环境
    set +u  # 关闭 unbound variable 报错（safe）
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate dnaberts
    
    echo "Python: $(which python)"
    python -c "import textaugment; print(textaugment.__file__)"


    # 运行脚本
    python $PROJECT_ROOT/scripts/myCluster_2.py \
      -vd "$vaef_file" \
      -ld "$LABEL_FILE" \
      -fd "$fasta_file" \
      -bd "$BINS_OUTPUT_DIR" \
      -cvf "$cvf_file"

    # 如果这是脚本里，运行完可选择退出环境
    conda deactivate

    
else
    echo "⏭️ 聚类已完成，跳过"
fi

# Step 9: CheckM2 评估
checkm_out="${output_dir}/${BASE_NAME}_checkm2"

checkm_db_dir="${HOME}/.checkm2/databases"  # ✅ 添加定义

if [ ! -f "$checkm_out/quality_report.tsv" ]; then
    echo "🔧 Step 9: CheckM2 评估..."
    #conda run -n checkm2 checkm2 predict -t 64 -x fasta --input "$BINS_OUTPUT_DIR" --output-directory "$checkm_out"
        
        # 激活 conda 环境
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate checkm2

    # 检查数据库是否存在
    if [ ! -d "$checkm_db_dir" ] || [ -z "$(ls -A "$checkm_db_dir")" ]; then
        echo "📦 未检测到 CheckM2 数据库，正在下载..."
        checkm2 database --download
    else
        echo "✅ CheckM2 数据库已存在，跳过下载"
    fi
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

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "✅ 所有流程结束，总运行时间: $runtime 秒"
