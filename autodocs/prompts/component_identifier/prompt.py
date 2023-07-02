ComponentIdentifierQAPrompt = """
Your task is to write a high level summary of a series of Machine Learning models and classes for a technical audience of data scientists.

The user will be provide a series of descriptions for {trace_type} functions that the user has run over the course of a python session, including
details about the arguments provided to those functions.

Summarise the functions into their base classes, i.e. MyClass.method() and MyClass.method2() becomes MyClass. Use a human readable
form of the class name, i.e. MyClass becomes My Class. 

For each class, write one of the three categories [Config, Processor, Model]. A Config is a class that contains
configuration information relating to the models. A Processor modifies the input data before it is provided to the model. 
A model outputs predictions given input data.

Provide the output in the following format:
[Category] (Item Name): Model Description
[Category] (Item Name): Model Description
...

Example input:
Name: optuna.trial.Trial.suggest_int
Description: This method of the Optuna Trial class provides an integer within the range 4 and 128, and names it "n_units_l"

Name: optuna.trial.Trial.suggest_float
Description: This method of the Optuna Trial class provides a float within the range 0.2 and 0.5, and names it "dropout"

Name: sklearn.preprocessing.StandardScaler.transform
Description: This method of the StandardScaler class performs standardization by centering and scaling.

Name: xgboost.XGBClassifier.predict
Description: This method of the XGBClassifier class predicts the class probability for each row input provided to it

Example output:
[Config] Optuna Trial: Provide random hyperparameters within a given range.
[Processor] StandardScaler: Standardize features by removing the mean and scaling to unit variance.
[Model] XGBClassifier: Predict the appropriate class.

Input:
{text}

Output:
"""


ComponentsRefinePrompt = """
Your task is to write a high level summary of a series of Machine Learning models and classes for a technical audience of data scientists.

The user will be provide a series of descriptions for {trace_type} functions that the user has run over the course of a python session, including
details about the arguments provided to those functions.

Summarise the functions into their base classes, i.e. MyClass.method() and MyClass.method2() becomes MyClass. Use a human readable
form of the class name, i.e. MyClass becomes My Class. 

For each class, write one of the three categories [Config, Processor, Model]. A Config is a class that contains
configuration information relating to the models. A Processor modifies the input data before it is provided to the model. 
A model outputs predictions given input data.

Provide the output in the following format:
[Category] (Item Name): Model Description
[Category] (Item Name): Model Description
...

Example input:
Name: optuna.trial.Trial.suggest_int
Description: This method of the Optuna Trial class provides an integer within the range 4 and 128, and names it "n_units_l"

Name: optuna.trial.Trial.suggest_float
Description: This method of the Optuna Trial class provides a float within the range 0.2 and 0.5, and names it "dropout"

Name: sklearn.preprocessing.StandardScaler.transform
Description: This method of the StandardScaler class performs standardization by centering and scaling.

Name: xgboost.XGBClassifier.predict
Description: This method of the XGBClassifier class predicts the class probability for each row input provided to it using a gradient boosted tree approach.

Example output:
[Config] Optuna Trial: Provide random hyperparameters within a given range.
[Processor] StandardScaler: Standardize features by removing the mean and scaling to unit variance.
[Model] XGBClassifier: Predict the appropriate class using a Gradient Boosted Tree Model from XgBoost.

We have already started the summary below;
{existing_answer}

There is also this additional context;
{text} 

Given the additional context, refine the original summary. 
If the context isn't useful, return the existing summary.
"""
