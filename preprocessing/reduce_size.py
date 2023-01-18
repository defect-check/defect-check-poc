from PIL import Image
import constants as C
import os.path as path


def reduce_size(filename):
    img = Image.open(filename)

    # Downsize the image with an ANTIALIAS filter (gives the highest quality)
    # Both Resnet-50 and VGG-16 use images of size 224 x 224
    # Fitting 9 of these for random crop, we have,
    img = img.resize((224 * 3, 224 * 3), Image.Resampling.LANCZOS)

    if path.isabs(filename) or path.normpath(filename).startswith(".."):
        filename = path.basename(filename)
    # Chose quality by trial and error to get 56kb images from 3.5MB sources
    img.save(path.join(C.SMALL_FOLDER, filename), optimize=True, quality=80)


if __name__ == "__main__":
    import sys

    files = sys.argv[1:]
    if len(files) == 1 and files[0].endswith(".txt"):
        # python preprocessing/reduce_size.py ../../DefectDetectData/Jpeg/all.txt
        dir = path.normpath(path.dirname(files[0]))
        with open(files[0]) as file:
            files = map(
                lambda x: path.abspath(path.join(dir, x.replace("\n", ""))),
                file.readlines(),
            )
    for arg in files:
        print(arg)
        reduce_size(arg)
