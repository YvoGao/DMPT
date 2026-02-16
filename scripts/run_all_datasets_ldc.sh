#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

bash scripts/beijing_opera.sh ldc
bash scripts/crema_d.sh ldc
bash scripts/esc50_actions.sh ldc
bash scripts/esc50.sh ldc
bash scripts/gt_music_genre.sh ldc
bash scripts/ns_instruments.sh ldc
bash scripts/ravdess.sh ldc
bash scripts/sesa.sh ldc
bash scripts/tut.sh ldc
bash scripts/urban_sound.sh ldc
bash scripts/vocal_sound.sh ldc