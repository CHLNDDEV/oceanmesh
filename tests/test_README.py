import os
import pathlib

this_dir = pathlib.Path(__file__).resolve().parent


def test_readme():

    os.system(f"pytest --codeblocks {this_dir.parent}/README.md")
