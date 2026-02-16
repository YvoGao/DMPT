#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

bash scripts/beijing_opera.sh moaalm
bash scripts/crema_d.sh moaalm
bash scripts/esc50_actions.sh moaalm
bash scripts/esc50.sh moaalm
bash scripts/gt_music_genre.sh moaalm
bash scripts/ns_instruments.sh moaalm
bash scripts/ravdess.sh moaalm
bash scripts/sesa.sh moaalm
bash scripts/tut.sh moaalm
bash scripts/urban_sound.sh moaalm
bash scripts/vocal_sound.sh moaalm