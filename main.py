import os
import argparse
from PIL import Image


def convert_to_png(source, destination=""):
    """
    converts an image into a png
    """
    if os.path.isfile(source):
        directory = os.path.dirname(source)
        file_name = os.path.basename(source)
        file_name = file_name.split(".")
        file_name[1] = '.png'
        file_name = "".join(file_name)
        
        img = Image.open(source)
        if destination != "" and os.path.isdir(destination):
            img.save(os.path.join(destination, file_name), "PNG" )
            return "image converted successfully with new destination directory"
        elif destination == "":
            img.save(os.path.join(directory, file_name), "PNG" )
            return "Image converted successfully"
        else:
            return "Please provide a valid destination directory"
    else:
        return "Please provide a valid path to your image"


def convert_to_jpg(source, destination=""):
    """
    converts an image into a jpg
    """
    if os.path.isfile(source):
        directory = os.path.dirname(source)
        file_name = os.path.basename(source)
        file_name = file_name.split(".")
        file_name[1] = '.jpg'
        file_name = "".join(file_name)
        img = Image.open(source)
        if destination != "" and os.path.isdir(destination):
            img.save(os.path.join(destination, file_name), "JPG")
        elif destination == "":
            img.save(os.path.join(directory, file_name), "JPG" )
        else:
            return "Please provide a valid destination directory"
    else:
        return "Please provide a valid path to your image"


def personal():
    """
    asdf,
    adsfasdf
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    subparsers = parser.add_subparsers(dest="commands")

    # convert subparser
    convert_subparser = subparsers.add_parser(
        name="convert", help="convert an image into another format"
    )
    convert_subparser.add_argument(
        "file_paths", type=str, nargs="*", help="an integer for the accumulator"
    )
    convert_subparser_mutually_exclusive = (
        convert_subparser.add_mutually_exclusive_group()
    )
    convert_subparser_mutually_exclusive.add_argument(
        "-p",
        "--png",
        action="store_true",
        help="convert an image into a PNG",
        default=False,
    )
    convert_subparser_mutually_exclusive.add_argument(
        "-j",
        "--jpg",
        action="store_true",
        help="convert an image into a JPG",
        default=False,
    )

    args = parser.parse_args()
    print(args.commands)
    if args.commands == "convert":
        length = len(args.file_paths)
        # print(args.file_paths)
        # print(args)

        if length <= 2:
            if length == 2:
                if args.jpg:
                    source, destination = args.file_paths
                    print(convert_to_jpg(source, destination))
                elif args.png:
                    source, destination = args.file_paths
                    print(convert_to_png(source, destination))
                else:
                    print(
                        "Please indicate whether you want to convert to a jpg with -j(or --jpg) or to a png with -p(or --png)"
                    )
            else:
                source = args.file_paths[0]
                if args.jpg:
                    print(convert_to_jpg(source))
                elif args.png:
                    print(convert_to_png(source))
                else:
                    print(
                        "Please indicate whether you want to convert to a jpg with -j(or --jpg) or to a png with -p(or --png)"
                    )
        else:
            print("The number of arguments given was too many")


if __name__ == "__main__":
    """
    The main function of the CLI,
    runs the argParse
    """
    personal()
