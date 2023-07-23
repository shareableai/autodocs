TracePrompt = """Could you please create a tree diagram, in indented list style, for the significant functions you 
used in your model? The diagram should start with the overall process at the root and branch off into the different 
stages, each of which can further branch off into specific functions or operations you used. Here's an example:


Root (Start of Process)
│
├── Data Cleaning
│   ├── Handling Missing Values
│   └── Outlier Detection and Handling
│
├── Exploratory Data Analysis
│   ├── Data Plotting
│   └── Correlation Calculation
│
└── Feature Engineering
    ├── Feature Creation
    └── Feature Scaling

In this example, each indentation level represents a level in the hierarchy, and the order of operations typically reads from top to bottom. Note: you should exclude irrelevant operations like lambda functions or minor helper functions; we're only interested in the major steps related to model training

## Functions
{text}

## Diagram
"""

TraceRefinePrompt = """Could you please continue and complete the tree diagram that I've started, based on the 
significant functions you used in your model? The diagram starts with the overall process at the root and branches 
off into the different stages. Each stage can further branch off into specific functions or operations you used.

Here's the part of the diagram I've started:

{existing_answer}

Please extend this diagram to include other stages or main steps you took, each with their specific functions or 
operations. Each indentation level represents a level in the hierarchy, and the order of operations should read from 
top to bottom.

Remember to exclude any operations that aren't directly related to model training, such as lambda functions or minor 
helper functions. We're primarily interested in the major steps and functions that contribute to understanding the 
overall modeling process. Thank you

## Functions
{text}

## Diagram
"""
