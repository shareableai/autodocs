import pathlib
from autodocs.document.doc import OutputDocumentation


def generate_html():
    OutputDocumentation(
        workspace_name='MyWorkspace',
        inference_steps={
            'Classification Model': 'This is a classification model for detecting fraud. It takes in a tabular dataset and outputs a float value per row that represents the likelihood of each row containing fraudulent behaviour.'
        },
        inference_hyperparams={
            'sklearn_model.break_split': True,
            'sklearn.my_model.mean': 3
        },
    ).to_html(pathlib.Path.cwd())


if __name__ == "__main__":
    generate_html()