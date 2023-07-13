import argparse
import pathlib
import uuid

from observer.tracking_type import TrackingType

from autodocs.document.doc import OutputDocumentation
from autodocs.trace import Trace, NoSuchTrace

parser = argparse.ArgumentParser(description='Generate Documentation for a Model')
parser.add_argument('trace_id', type=str,
                    help='Trace Identifier from Observer')

if __name__ == "__main__":
    args = parser.parse_args()
    trace_id = args.trace_id
    (pathlib.Path.home() / ".stack_traces").mkdir(exist_ok=True)
    (pathlib.Path.home() / ".autodocs").mkdir(exist_ok=True)
    autodocs_dir = pathlib.Path.home() / ".autodocs" / trace_id
    autodocs_dir.mkdir(exist_ok=True, parents=True)
    output_dir = pathlib.Path.cwd() / 'documentation_output'
    for segment_name in [TrackingType.Processing, TrackingType.Training, TrackingType.Inference]:
        try:
            trace = Trace.from_id(
                uuid.UUID(trace_id), segment_name
            )
            trace.save_hyperparameters()
            trace.describe_trace()
            trace.trace_graph()
            trace.describe_classes()
        except NoSuchTrace:
            pass
    output_doc = OutputDocumentation.from_autodocs_directory(autodocs_dir)
    output_doc.to_html(output_dir)
    print(f"Created HTML output at {list(output_dir.glob('*.html'))}")
