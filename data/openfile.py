# %%
import tarfile

# open file
with tarfile.open("imagenette2.tgz") as file:
    file.extractall(".")
