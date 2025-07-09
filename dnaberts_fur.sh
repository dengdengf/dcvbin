#!/bin/bash
#SBATCH --job-name=dnaberts_fur           # 作业名称
#SBATCH --output=dnaberts_fur_output.txt        # 输出日志的文件名
#SBATCH --time=120:00:00            # 执行时间限制为1小时
#SBATCH --ntasks=1                 # 任务数为1
#SBATCH --cpus-per-task=48         # 每个任务使用2个 CPU 核心
#SBATCH --mem=256G                   # 每个任务使用4G内存
#SBATCH --partition=gpujl     # 队列名称为gpujl
#SBATCH --gres=gpu:4               # 如果需要，使用1个GPU



unset LD_LIBRARY_PATH

# === 正确加载 conda 环境 ===
__conda_setup="$('/home/wangjingyuan/anaconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    source /home/wangjingyuan/anaconda3/etc/profile.d/conda.sh
fi
unset __conda_setup

# === 激活你需要的环境===
conda activate vambnew


bash /home/wangjingyuan/lyf/DCVBin_project/bin/my_method.sh /home/wangjingyuan/lyf/DCVBin_project/test_zhenshi_fur /home/wangjingyuan/lyf/DCVBin_project/test_zhenshi_fur/output