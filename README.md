personal code and scripts for deep learning work


## Structure

Code is located in `model-eval` and scripts to invoke the code are in `scripts`. Models are located in `model-eval/models_implementation` and that is where most the work is in -- importing a pre-defined model and tweaking parts of it.


## Usage

The way I use it most of the time is creating a TOML document queueing up a few experiments of similar types with slightly different parameters, and invoking `scripts/train2.ipy` on it.
