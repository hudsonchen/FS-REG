#!/bin/bash -l

# Batch script to run a hybrid parallel job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:10:0

# Request 1 gigabyte of RAM per core (must be an integer)
#$ -l mem=1G

# Request 15 gigabytes of TMPDIR space per node (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N MadIntelHybrid

# Select the MPI parallel environment and 80 cores.
#$ -pe mpi 80

# Set the working directory to somewhere in your scratch space.
# This directory must exist.
#$ -wd /home/ucabzc9/scratch/output/

nvidia-smi
