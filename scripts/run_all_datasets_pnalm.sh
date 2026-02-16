#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

bash scripts/beijing_opera.sh pnalm
bash scripts/crema_d.sh pnalm
bash scripts/esc50_actions.sh pnalm
bash scripts/esc50.sh pnalm
bash scripts/gt_music_genre.sh pnalm
bash scripts/ns_instruments.sh pnalm
bash scripts/ravdess.sh pnalm
bash scripts/sesa.sh pnalm
bash scripts/tut.sh pnalm
bash scripts/urban_sound.sh pnalm
bash scripts/vocal_sound.sh pnalm