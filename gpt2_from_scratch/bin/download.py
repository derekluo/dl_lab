#!/usr/bin/env python3

import requests
import glob
import os

def download_file(url: str, output_path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
def find_txt_files(directory: str):
    return glob.glob(os.path.join(directory, "**","*.txt"), recursive=True)

def concatenate_txt_files(files: list[str], output_file: str):
    with open(output_file, "w") as f:
        for file in files:
            with open(file, "r") as in_file:
                f.write(in_file.read() + "\n")
                
directory = "data"
input_dir = os.path.join(directory, "scifi")
output_file = os.path.join(directory, "scifi.txt")

txt_files = find_txt_files(input_dir)
concatenate_txt_files(txt_files, output_file)