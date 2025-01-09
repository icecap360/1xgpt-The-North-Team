#!/bin/bash
output_dir='/pub0/qasim/1xgpt/1xgpt/outputs/generated/baseline_35M'
checkpoint_dir='checkpoints/GENIE_35M'
for i in {0..240..10}; do
    python genie/generate.py --checkpoint_dir $checkpoint_dir \
        --output_dir $output_dir --example_ind $i --maskgit_steps 2 --temperature 0
    python visualize.py --token_dir $output_dir
    mv $output_dir/generated_offset0.gif $output_dir/example_$i.gif
    mv $output_dir/generated_comic_offset0.png $output_dir/example_$i.png
done

# Evaluate
python genie/evaluate.py --checkpoint_dir $checkpoint_dir --maskgit_steps 2