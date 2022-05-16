#!/bin/bash
#SBATCH -J main
#SBATCH -p CS282
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=yangqy@shanghaitech.edu.cn
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

cd /public/home/yangqy/nocall
python main.py > output.out
