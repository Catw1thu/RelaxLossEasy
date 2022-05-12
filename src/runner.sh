echo 'Calling scripts!'

# # toggle this on if you want to delete old logs each time you run new experiments
# rm -rf ../logs
# rm -rf ../plots
# mkdir ../plots

python main.py &

python main.py --relax_alpha=0.5 --device=cuda:1 &
python main.py --relax_alpha=1.0 --device=cuda:1



echo 'All experiments are finished!'