#!/bin/sh

# 定义后缀变量
LOG_SUFFIX="m1"

export PATH=/root/miniconda3/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0
eval "$(conda shell.bash hook)"
conda activate saidi

# 使用变量生成日志文件
# nohup sh saidi_scripts/TSMixer_saidi.sh > saidi_log/TSMixer${LOG_SUFFIX}.log 2>&1 &
# nohup sh saidi_scripts/TimeMixer_saidi.sh > saidi_log/TimeMixer${LOG_SUFFIX}.log 2>&1 &
# nohup sh saidi_scripts/TimesNet_saidi.sh > saidi_log/TimesNet${LOG_SUFFIX}.log 2>&1 &
# nohup sh saidi_scripts/PatchTST_saidi.sh > saidi_log/PatchTST${LOG_SUFFIX}.log 2>&1 &
# nohup sh saidi_scripts/Informer_saidi.sh > saidi_log/Informer${LOG_SUFFIX}.log 2>&1 &
# nohup sh saidi_scripts/Crossformer_saidi.sh > saidi_log/Crossformer${LOG_SUFFIX}.log 2>&1 &
nohup sh saidi_scripts/Autoformer_saidi.sh > saidi_log/Autoformer${LOG_SUFFIX}.log 2>&1 &
