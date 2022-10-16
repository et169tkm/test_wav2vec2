#!/bin/bash

for name in *.mp3; do
  nn=$( echo "$name" | sed 's/mp3/wav/' )
  if -e "$nn"; then
    echo "Skipping file $name"
    continue
  fi
  ffmpeg -i "$name" -map_channel 0.0.0 -ar 16000 "$nn"
done

