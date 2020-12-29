import pathlib
import zipfile

import exdown
import pytest
import requests

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "string,lineno",
    exdown.extract(
        this_dir.parent / "README.md", syntax_filter="python", max_num_lines=100000
    ),
)
def test_readme(string, lineno):

    # download
    url = "http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip"
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # un-compress
    with zipfile.ZipFile("gshhg-shp-2.3.7.zip", "r") as zip_ref:
        zip_ref.extractall("gshhg-shp-2.3.7")

    try:
        # https://stackoverflow.com/a/62851176/353337
        exec(string, {"__MODULE__": "__main__"})
    except Exception:
        print(f"README.md (line {lineno}):\n```\n{string}```")
        raise
