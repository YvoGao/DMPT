#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

bash scripts/beijing_opera.sh claps
bash scripts/crema_d.sh claps
bash scripts/esc50_actions.sh claps
bash scripts/esc50.sh claps
bash scripts/gt_music_genre.sh claps
bash scripts/ns_instruments.sh claps
bash scripts/ravdess.sh claps
bash scripts/sesa.sh claps
bash scripts/tut.sh claps
bash scripts/urban_sound.sh claps
bash scripts/vocal_sound.sh claps