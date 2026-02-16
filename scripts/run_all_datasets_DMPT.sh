#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

bash scripts/beijing_opera.sh DMPT
bash scripts/crema_d.sh DMPT
bash scripts/esc50_actions.sh DMPT
bash scripts/esc50.sh DMPT
bash scripts/gt_music_genre.sh DMPT
bash scripts/ns_instruments.sh DMPT
bash scripts/ravdess.sh DMPT
bash scripts/sesa.sh DMPT
bash scripts/tut.sh DMPT
bash scripts/urban_sound.sh DMPT
bash scripts/vocal_sound.sh DMPT