import os

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(current_dir + "/../")


def test_readme():

    os.system(f"pytest --codeblocks {parent_dir}/README.md")
