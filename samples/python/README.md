## How to make and use a conda/Python environment 
This is how the file conda_spec_platforms.txt was created on a machine that was running in the conda environemt
```
conda list --explicit > conda_spec_platforms.txt

```

To make a conda environment consisting packages in this list we do:
```
module load miniconda
conda create --name platforms --file path/conda_spec_platforms.txt
source activate platforms
```
