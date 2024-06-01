import os
import argparse
from PIL import Image
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bright_green",
})
console = Console(theme=custom_theme)


def convert_to_png(source, destination=""):
    """
    converts an image into a png
    """
    # print(os.path.abspath(source))
    source = os.path.abspath(source)
    if os.path.isfile(source):
        directory = os.path.dirname(source)
        file_name = os.path.basename(source)
        file_name = file_name.split(".")
        file_name[1] = '.png'
        file_name = "".join(file_name)
        
        try:
            img = Image.open(source)
        except:
             console.print("There was an error trying to open the image", style="danger")
             return
        if destination != "" and os.path.isdir(destination):
            try:
                img.save(os.path.join(destination, file_name), "PNG" )
            except:
                console.print("There was an error converrting the image to a PNG", style="danger")
                
            console.print("image converted successfully with new destination directory", style="success")
        elif destination == "":
            try:
                img.save(os.path.join(directory, file_name), "PNG" )
            except:
                console.print("There was an error converting the image to a PNG", style="danger")
                
            console.print("Image converted successfully", style="success")
        else:
            console.print("Please provide a valid destination directory", style="danger")
    else:
         console.print("Please provide a valid path to your image", style="danger")
         return



def convert_to_jpg(source, destination=""):
    """
    converts an image into a jpg
    """
    source = os.path.abspath(source)
    if os.path.isfile(source):
        directory = os.path.dirname(source)
        file_name = os.path.basename(source)
        file_name = file_name.split(".")
        file_name[1] = '.jpg'
        file_name = "".join(file_name)
        try:
            img = Image.open(source)
        except:
            console.print("There was an error trying to open the image", style="danger")
            return
        if destination != "" and os.path.isdir(destination):
            try:
                img.save(os.path.join(destination, file_name), "JPG")
            except:
                console.print("There was an error converrting the image to a JPG", style="danger")
                return
        elif destination == "":
            try:
                img.save(os.path.join(directory, file_name), "JPG" )
            except:
                console.print("There was an error converrting the image to a JPG", style="danger")
                return
        else:
            console.print("Please provide a valid destination directory", style="danger")
            return
    else:
        console.print("Please provide a valid path to your image", style="danger")
        return


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
    if args.commands == "convert":
        length = len(args.file_paths)

        if length <= 2:
            if length == 2:
                if args.jpg:
                    source, destination = args.file_paths
                    convert_to_jpg(source, destination)
                elif args.png:
                    source, destination = args.file_paths
                    convert_to_png(source, destination)
                else:
                    console.print(
                        "Please indicate whether you want to convert to a jpg with -j(or --jpg) or to a png with -p(or --png)", style="danger"
                    )
            else:
                source = args.file_paths[0]
                if args.jpg:
                    convert_to_jpg(source)
                elif args.png:
                    convert_to_png(source)
                else:
                    console.print(
                        "Please indicate whether you want to convert to a jpg with -j(or --jpg) or to a png with -p(or --png)", style="danger"
                    )
        else:
            console.print("The number of arguments given was too many", style="danger")


if __name__ == "__main__":
    """
    The main function of the CLI,
    runs the argParse
    """
    personal()
