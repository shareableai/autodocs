FunctionParameterPrompt = """As a Data Scientist, it's essential to provide comprehensive descriptions for each function 
parameter. To ensure accuracy, consider whether each parameter acts as a hyperparameter in the machine learning 
context. A hyperparameter is a tunable parameter, not determined by the data, that directly impacts the behavior of a 
machine learning model, excluding purely aesthetic or logging behaviour. Changing hyperparameters can influence the 
model's structure, regularization, learning rates, and batch sizes. 

If the function has no parameters, simply respond with "There are no parameters." If there are parameters, 
provide a concise breakdown for each one, stating its name and your reasoning for classifying it as a hyperparameter 
or not. Keep to the following format: "- [PARAMETER_NAME]: [HYPERPARAMETER/NOT HYPERPARAMETER], [REASONING]". Avoid 
adding extraneous details.

When the parameter resides in the self or cls object, use the notation self.parameter or cls.parameter and list it 
separately from the self parameter. If it doesn't, represent it in its regular form.

Input:
{text}

Parameters:
"""

FunctionRefinePrompt = """As a Data Scientist, it's essential to provide comprehensive descriptions for each function 
parameter. To ensure accuracy, consider whether each parameter acts as a hyperparameter in the machine learning 
context. A hyperparameter is a tunable parameter, not determined by the data, that directly impacts the behavior of a 
machine learning model, excluding purely aesthetic or logging behaviour. Changing hyperparameters can influence the 
model's structure, regularization, learning rates, and batch sizes. 

If the function has no parameters, simply respond with "There are no parameters." If there are parameters, 
provide a concise breakdown for each one, stating its name and your reasoning for classifying it as a hyperparameter 
or not. Keep to the following format: "- [PARAMETER_NAME]: [HYPERPARAMETER/NOT HYPERPARAMETER], [REASONING]". Avoid 
adding extraneous details.

When the parameter resides in the self or cls object, use the notation self.parameter or cls.parameter and list it 
separately from the self parameter. If it doesn't, represent it in its regular form.

Input:
{text}

We have provided an existing summary up to a certain point.

Existing Parameters:
{existing_answer}

We have the opportunity to refine the existing summary (only if needed) with some more context below. 

Context:
{text} 

Given the new context, refine the original description. 
If the context isn't useful, return the existing description.
If no refinements are necessary, return the existing description.

Parameters:
"""