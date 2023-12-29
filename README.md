# LLM Annotation Realiablity

This project to verify the reliability of the annotation from LLMs on the preference data.
Recently, resesarchers have proposed to use LLMs to generate preference data for training preference-based RL algorithms, e.g. the RLAIF paper.
However, the reliability of the AI feedback is not clear.
This project aims to verify the reliability of the AI feedback from LLMs.

To do so, we use the preference data from human annotators in the OpenAI's TL;DR dataset, and the preference annotated by LLMs for the same set of prompts and responses.
We then compare the two sets of preference data to see if they are consistent through the correlation cofficient.

## Architecture

The project is organized as follows:

`main.py`: the main script to run the project.
`data.py`: the script to load the data.
`prompts.py`: the script to configurate the prompts.
`annotate.py`: the script to annotate the data by AI.
`utils.py`: the script to provide utility functions.
