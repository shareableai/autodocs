FunctionParameterPrompt = """You are a Data Scientist writing a summary for each parameter passed to a function.

Think carefully, making sure your answers are correct.

For each parameter, write your reasoning as to whether it is a hyperparameter, where a hyperparameter is a configurable 
parameter that is external to the data and directly influences the behaviour of a machine learning model. 
Hyperparameters can be adjusted or tuned to optimize model performance, affecting aspects such as model 
architecture, regularization, learning rates, and batch sizes.

For each parameter, write the parameter name and your reasoning for whether it is or isn't a hyperparameter. If a parameter is on the `self` or `cls` object, 
write it as `self.parameter` or `cls.parameter` on a separate line to the self parameter. Otherwise, write it out 
normally.

Input:
{text}

Summary:
"""

FunctionRefinePrompt = """You are a Data Scientist writing a summary for each parameter passed to a function.

Think carefully, making sure your answers are correct.

For each parameter, determine if it is a hyperparameter, where a hyperparameter is a configurable 
parameter that is external to the data and directly influences the behaviour of a machine learning model. 
Hyperparameters can be adjusted or tuned to optimize model performance, affecting aspects such as model 
architecture, regularization, learning rates, and batch sizes.

For each parameter, write the parameter name and your reasoning for whether it is or isn't a hyperparameter. If a parameter is on the `self` or `cls` object, 
write it as `self.parameter` or `cls.parameter` on a separate line to the self parameter. Otherwise, write it out 
normally.

We have provided an existing summary up to a certain point.

Existing Summary:
{existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

Context:
{text} 

Given the new context, refine the original description. 
If the context isn't useful, return the existing description.
If no refinements are necessary, return the existing description.

Summary:
"""