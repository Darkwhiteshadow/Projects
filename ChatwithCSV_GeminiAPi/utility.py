import textwrap
import time
import re
import google.generativeai as genai
from IPython.display import Markdown


def to_markdown(text):
    text = text.replace('.', '*')
    return Markdown(textwrap.indent(text, '>', predicate=lambda _: True))


def upload_file(path, mime_type='text/csv'):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def check_activation_of_file(files):
    print("file is processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")


def get_code(input_string):
    pattern = r"```python\s*(.*?)\s*```"

    code = re.findall(pattern, input_string, re.DOTALL)
    return code
