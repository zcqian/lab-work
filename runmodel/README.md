# RunModel 

This is a library/framework for running and evaluating PyTorch models.

## Design Documentation

Before I start coding this thing, I will write the documentation first to get a more clear view on what features are needed, how each part should fit together, and how everything should work. Then the work will be implementing this thing.

## Motives

The 0th stage of writing code for these tasks is writing things and duplicating the script and making modification. The 1st stage would be actually trying to reuse some parts of the code, by moving some of the code to functions, and trying to reuse them.

However, models change more often than the code used to train models, so I moved on to use a script where models are dynamically loaded, but the process of training and evaluating is reused. This is what I define as the 2nd stage, which is the state of things before I started RunModel. This scheme of doing things also has its disadvantages. One thing is that models may require parameters. I used to have a "model generator" and the main script would pass parameters to the generator. However, most of the times this is unnecessary and it would also be difficult to track these parameters down. Therefore, I moved to a scheme where the main script does not pass parameters to the model, but the model/optimizer all have distinctive names, and the results are store in various paths, reflecting the configuration it used. But, this does still seem hacky.

The other issue with the "main script dynamically load models" scheme is that sometimes I need to override the behaviors of the main script, for instance, at this time I have the need to alter what parts of the model to fine tune in different epochs. This isn't possible without lots of monkey-patching or other dirty hacks. And this adds on top of how each "model" is basically doing monkey patching on top of more monkey patched models. The models code is still readable (cleaner than not doing patching), but messing with the script will not be clean.

This means its time for another scheme where there should be less repetition and less dirty hacks, while also being more versatile to accomplish my tasks.

The basic plan is to either implement a framework where the process is much more fine-grained than the previous "main.py" script, and I can override behaviors of each step very easily. Or make it possible to add hooks to every step of the process.
