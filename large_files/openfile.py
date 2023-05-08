# %%
import tarfile

# open file
file = tarfile.open("imagenette2.tgz")


# extracting file
file.extractall("./Destination_FolderName")

file.close()
