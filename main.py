import os
import argparse
from PIL import Image


def convert_to_png(source, destination):
    if os.path.exists(source):
        return True
    else:
        return False

def personal():
    """
    asdf,
    adsfasdf
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    subparsers = parser.add_subparsers(dest="commands")
    convert_subparser = subparsers.add_parser(name="convert",description="convert an image into another format")
    convert_subparser.add_argument('file_paths',type=str, nargs='*',
                    help='an integer for the accumulator')
    

    args = parser.parse_args()
    print(args.commands)
    if args.commands == "convert":
        if len(args.file_paths) <= 2:
            source, destination = args.file_paths
            print(convert_to_png(source, destination))
            # print(f"{source=}, {destination=}")
        else:
            print("The number of arguments given was too many")

if __name__ == "__main__":
    """
    The main function of the CLI,
    runs the argParse
    """
    personal()
    
