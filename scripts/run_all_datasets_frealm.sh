#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

bash scripts/beijing_opera.sh frealm
bash scripts/crema_d.sh frealm
bash scripts/esc50_actions.sh frealm
bash scripts/esc50.sh frealm
bash scripts/gt_music_genre.sh frealm
bash scripts/ns_instruments.sh frealm
bash scripts/ravdess.sh frealm
bash scripts/sesa.sh frealm
bash scripts/tut.sh frealm
bash scripts/urban_sound.sh frealm
bash scripts/vocal_sound.sh frealm