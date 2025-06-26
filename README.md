This repository contains numerical experiment codes for *Spatial QUBO: Convolutional Formulation of Large-Scale Binary Optimization with Dense Interactions*.

It also includes `spqubolib`, a Python library 
for constructing and solving optimization problems using spatial QUBO (spQUBO).


# Requirements

This project was developed under Python 3.11.12 and other modules.

For a complete list of project requirements, please refer to the `Dockerfile`.

# Usage

- First, compile the C++ module in `spqubolib` using the following commands:

```bash
cd spqubolib/spqubolib
python setup.py build_ext --inplace
```

- Execute the code in `placement/placement.ipynb`.
  - You can run specific parts by configuring `flag_plot`, `flag_run`, and `flag_comptime`.
- Execute the code in `clustering/blob.ipynb`.
  - You can run specific parts by configuring `flag_plot` and `flag_run`.

For each experiment in `clustering` or `placement`:

- Output data are stored in the `results` directory.
- Result plots are saved in the `images` directory.
- For the placement problem, computation time measurements are stored in the `comptime` directory.

## Run Pipeline 

You can also reproduce the overall experiment with the following command:

```
dvc repro
```

- To run this command, `dvc` must be installed.

## docker-compose

You can run the program in a Docker container.
Use the provided `docker-compose.yml` to launch the container and execute the commands or notebooks within it.

