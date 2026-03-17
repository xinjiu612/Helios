#!/bin/bash

set -euo pipefail

SCRIPT="/beijing-c/workspace/zhumo/diff/task_design/language_control/i2v_base_fast_move_motion.py"
PROMPT_DIR="/beijing-c/workspace/zhumo/diff/task_design/language_control/prompts"
PY="/beijing-c/workspace/zhumo/miniconda3/envs/diffusion/bin/python"

PROMPTS=(
  "00_strong_push_office.txt"
  "01_left_tree_truckin.txt"
  "02_official_fish_dynamic.txt"
  "03_official_red_balloon_tracking.txt"
  "04_official_suv_following_shot.txt"
  "05_official_drone_circle_church.txt"
  "06_official_train_timelapse.txt"
  "07_official_octopus_crab.txt"
)

for i in "${!PROMPTS[@]}"; do
  gpu="$i"
  prompt_file="${PROMPT_DIR}/${PROMPTS[$i]}"
  tag="base_g${gpu}_p${i}"
  echo "Launching GPU ${gpu} with ${prompt_file} -> ${tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" "$PY" "$SCRIPT" \
    --prompt_file "$prompt_file" \
    --tag "$tag" \
    --seed "$((142 + i))" &
done

wait
echo "Base 8-GPU prompt sweep finished."
#!/bin/bash

set -euo pipefail

SCRIPT="/beijing-c/workspace/zhumo/diff/task_design/language_control/i2v_base_fast_move_motion.py"
PROMPT_DIR="/beijing-c/workspace/zhumo/diff/task_design/language_control/prompts"
PY="/beijing-c/workspace/zhumo/miniconda3/envs/diffusion/bin/python"

PROMPTS=(
  "00_strong_push_office.txt"
  "01_left_tree_truckin.txt"
  "02_official_fish_dynamic.txt"
  "03_official_red_balloon_tracking.txt"
  "04_official_suv_following_shot.txt"
  "05_official_drone_circle_church.txt"
  "06_official_train_timelapse.txt"
  "07_official_octopus_crab.txt"
)

for i in "${!PROMPTS[@]}"; do
  gpu="$i"
  prompt_file="${PROMPT_DIR}/${PROMPTS[$i]}"
  tag="base_g${gpu}_p${i}"
  echo "Launching GPU ${gpu} with ${prompt_file} -> ${tag}"

  CUDA_VISIBLE_DEVICES="${gpu}" "$PY" "$SCRIPT" \
    --prompt_file "$prompt_file" \
    --tag "$tag" \
    --seed "$((100 + i))" &
done

wait
echo "Base 8-GPU prompt sweep finished."


