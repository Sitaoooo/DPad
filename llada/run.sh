# task=gsm8k-split
# model=instruct
task=gsm8k
task=humaneval
model=1.5
p=out
shot=5
length=256
block_length=32
threshold=0.9
sigma=3
scale=1.1
name=test

#parallel
steps=$((length / block_length))
#top-k
steps=${length}

#top-k
python run.py --task=${task} --model=${model} --num_fewshot=${shot} --length=${length} --block_length=${block_length} --steps=${steps} --sigma=${sigma} --scale=${scale} -re
#parallel
python run.py --task=${task} --model=${model} --num_fewshot=${shot} --length=${length} --block_length=${block_length} --steps=${steps} --threshold=0.9 --sigma=${sigma} --scale=${scale} -re 
#parallel+prefix
python run.py --task=${task} --model=${model} --num_fewshot=${shot} --length=${length} --block_length=${block_length} --steps=${steps} --threshold=0.9 --sigma=${sigma} --scale=${scale} -re -c
#parallel+dual
python run.py --task=${task} --model=${model} --num_fewshot=${shot} --length=${length} --block_length=${block_length} --steps=${steps} --threshold=0.9 --sigma=${sigma} --scale=${scale} -re -dc

python run.py --task=gsm8k-split --model=instruct --num_fewshot=5 --length=256 --block_length=32 --steps=8 --threshold=0.9 --sigma=4 --scale=2 -re -c

python run.py --task=mbpp --model=1.5 --num_fewshot=3 --length=512 --block_length=32 --steps=8 --threshold=0.9 --sigma=3 --scale=1.6 --window=256 -re -c

python run.py -t=gsm8k-split -m=instruct -s=4 -l=256 -b=32 -th=0.9 -d=gaussian -k=4 -sc=2 -re -c -e
