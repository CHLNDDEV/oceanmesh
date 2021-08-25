import os
import pathlib
import zipfile
import requests

this_dir = pathlib.Path(__file__).resolve().parent


def test_readme():

    # download
    url = "http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip"
    filename = url.split("/")[-1]
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # un-compress
    with zipfile.ZipFile("gshhg-shp-2.3.7.zip", "r") as zip_ref:
        zip_ref.extractall("gshhg-shp-2.3.7")

    print(pathlib.Path("gshhg-shp-2.3.7").resolve())

    os.system(f"pytest --codeblocks {this_dir.parent}/README.md")
