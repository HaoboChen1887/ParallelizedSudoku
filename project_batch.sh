#!/bin/bash -
#===============================================================================
#
#          FILE: 1000_hw4_batch_script.sh
#
#         USAGE: ./1000_hw4_batch_script.sh
#
#   DESCRIPTION: 
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 11/18/19 17:37:34
#      REVISION:  ---
#===============================================================================

#SBATCH --job-name=proj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f19_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

./main 9 > output.txt

