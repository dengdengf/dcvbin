#!/bin/bash
#SBATCH --job-name=dnaberts
#SBATCH --output=dnaberts_output.txt
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:4

# === 干净地初始化 conda 环境 ===
unset LD_LIBRARY_PATH  # 避免系统 bash 被 anaconda 的库污染

source /home/wangjingyuan/anaconda3/etc/profile.d/conda.sh
conda activate vambnew

# === 输出当前环境检查 ===
echo "✅ Conda path: $(which conda)"
echo "✅ Python path: $(which python)"
echo "✅ 当前环境: $(conda info --envs | grep '*' )"

# === 启动主任务 ===
bash /home/wangjingyuan/lyf/DCVBin_project/bin/my_method.sh \ /home/wangjingyuan/lyf/DCVBin_project/test_zhenshi \ /home/wangjingyuan/lyf/DCVBin_project/test_zhenshi/output
