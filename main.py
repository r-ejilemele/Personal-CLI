import os
import argparse
import winreg as reg
import ctypes
from ctypes import wintypes
from PIL import Image
from rich.console import Console
from rich.theme import Theme
from rich.progress import track
from rich.progress import Progress, TextColumn, BarColumn
import psutil
import scraper
import compress

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red",
        "success": "bright_green",
    }
)
console = Console(theme=custom_theme)


class SHQUERYRBINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.ULONG),
        ("i64Size", ctypes.c_longlong),
        ("i64NumItems", ctypes.c_longlong),
    ]


def is_trash_empty():
    # recycle_bin_path = "C:\\$Recycle.Bin"
    # content = os.listdir(recycle_bin_path)
    # print(content)
    # return len(content) == 0
    # Load necessary Windows DLLs
    shell = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32

    info = SHQUERYRBINFO()
    info.cbSize = ctypes.sizeof(info)

    # Initialize COM library
    ole32.CoInitialize(0)

    try:
        result = shell.SHQueryRecycleBinW(None, ctypes.byref(info))
        if result == 0:
            return info.i64NumItems == 0
        else:
            console.print(f"The error {result} occurred.", style="danger")
            return False
    except Exception as e:
        console.print(
            "There was an error checking if the trash was empty", style="danger"
        )
    finally:
        # Uninitialize COM library
        ole32.CoUninitialize()


def empty_trash():
    # Define constants from the Windows API
    if is_trash_empty():
        console.print("The trash is already empty", style="success")
    else:
        SHEmptyRecycleBin = ctypes.windll.shell32.SHEmptyRecycleBinW
        bin_flags = 0  # 0 for all items in recycle bin, or use appropriate flags

        # Call the Windows API function to empty the recycle bin
        result = SHEmptyRecycleBin(None, None, bin_flags)

        if result == 0:
            console.print("Recycle bin emptied successfully.", style="success")
        else:
            console.print(
                f"Failed to empty recycle bin. Error code:\n {result}", style="danger"
            )


def get_dark_mode():
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    value_name = "AppsUseLightTheme"

    try:
        registry_key = reg.OpenKey(reg.HKEY_CURRENT_USER, key_path, 0, reg.KEY_READ)

        value, _ = reg.QueryValueEx(registry_key, value_name)

        reg.CloseKey(registry_key)

        dark_mode = value == 0
        return dark_mode

    except Exception as e:
        console.print(f"Failed to get dark mode:\n {e}", style="danger")
        return None


def toggle_dark():
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    value_name = "AppsUseLightTheme"
    current_mode = get_dark_mode()
    if current_mode is not None:
        try:
            registry_key = reg.OpenKey(
                reg.HKEY_CURRENT_USER, key_path, 0, reg.KEY_WRITE
            )
            reg.SetValueEx(
                registry_key,
                value_name,
                0,
                reg.REG_DWORD,
                0 if int(not (current_mode)) else 1,
            )
            reg.CloseKey(registry_key)
            console.print(
                f"Set {'dark' if int(not(current_mode)) else 'light'} mode successfully.",
                style="success",
            )
        except Exception as e:
            console.print(f"Failed to set dark mode: {e}", style="danger")
    else:
        console.print(
            "There was an error retrieving dark mode status from the system",
            style="danger",
        )


def battery():
    # returns a tuple

    laptop_battery = psutil.sensors_battery()
    battery_percentage = laptop_battery.percent

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green"),
        TextColumn("[green]{task.percentage:>3.0f}%"),
    ) as progress:

        task1 = progress.add_task("[blue]Battery Percentage", total=100)

        while not progress.finished:
            progress.update(task1, advance=battery_percentage)
            break


def gather(source):
    """
    Gathers all JPGs into in a directory into a folder
    """
    if os.path.isdir(source):
        new_dir_path = os.path.join(source, "JPGs")
        os.makedirs(os.path.join(source, "JPGs"))
        try:
            for file in os.listdir(source):
                if os.path.splitext(file)[1].lower() == ".jpg":
                    file_name = os.path.basename(file)
                    os.rename(
                        os.path.join(source, file_name),
                        os.path.join(new_dir_path, file_name),
                    )
            console.print("All files moved successfully!", style="success")
        except Exception as e:
            console.print(f"There was an error moving a file, \n {e}", style="danger")
    else:
        console.print("This directory does not exist", style="danger")


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
        file_name[1] = ".png"
        file_name = "".join(file_name)

        try:
            img = Image.open(source)
        except Exception as e:
            console.print(
                f"There was an error trying to open the image: \n {e}", style="danger"
            )
            return
        if destination != "" and os.path.isdir(destination):
            try:
                img.save(os.path.join(destination, file_name), "PNG")
            except Exception as e:
                console.print(
                    f"There was an error saving the image as a PNG: \n {e}",
                    style="danger",
                )

            console.print(
                "image converted successfully with new destination directory",
                style="success",
            )
        elif destination == "":
            try:
                img.save(os.path.join(directory, file_name), "PNG")
            except Exception as e:
                console.print(
                    f"There was an error saving the image as a PNG: {e}", style="danger"
                )

            console.print("Image converted successfully", style="success")
        else:
            console.print(
                "Please provide a valid destination directory", style="danger"
            )
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
        file_name[1] = ".jpg"
        file_name = "".join(file_name)
        try:
            img = Image.open(source)
        except Exception as e:
            console.print(
                f"There was an error trying to open the image: \n {e}", style="danger"
            )
            return
        if destination != "" and os.path.isdir(destination):
            try:
                img.save(os.path.join(destination, file_name), "JPEG")
            except Exception as e:
                console.print(
                    f"There was an error saving the image as a JPEG: \n {e}",
                    style="danger",
                )
                return
        elif destination == "":
            try:
                img.save(os.path.join(directory, file_name), "JPEG")
            except Exception as e:
                console.print(
                    f"There was an error saving the image as a JPEG: {e}", style="danger"
                )
                return
        else:
            console.print(
                "Please provide a valid destination directory", style="danger"
            )
            return
    else:
        console.print("Please provide a valid path to your image", style="danger")
        return


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def personal():
    """
    The main body of the personal CLI

    This is where all the parse arguments and parsers
        are added
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    subparsers = parser.add_subparsers(dest="commands")

    # convert subparser
    convert_subparser = subparsers.add_parser(
        name="convert", help="convert an image into another format"
    )
    convert_subparser.add_argument("file_paths", type=str, nargs="*", help="")
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
        help="convert an image into a JPEG",
        default=False,
    )

    # gather subparser
    gather_subparser = subparsers.add_parser(
        name="gather", help="gather all the JPEGs in a directory into one folder"
    )

    gather_subparser.add_argument(
        "file_path",
        type=str,
        nargs="?",
    )

    # battery subparse
    battery_subparser = subparsers.add_parser(
        name="battery", help="convert an image into another format"
    )
    dark_subparser = subparsers.add_parser(
        name="dark", help="toggle dark mode for my computer"
    )
    empty_trash_subparser = subparsers.add_parser(
        name="empty", help="toggle dark mode for my computer"
    )
    scrape_for_news = subparsers.add_parser(
        name="scrape", help="get most important news for the day"
    )
    compress_image = subparsers.add_parser(
        name="compress", help="Compress an image using using svd or pca"
    )
    compress_image.add_argument(
        "image_path", type=str, help="the file path to the image you want to compress"
    )
    compress_image.add_argument(
        "compression_level",
        type=int,
        help="Choose a compression level between 1 and 100 for your image",
        default=0,
    )
    compress_mutually_exclusiv = compress_image.add_mutually_exclusive_group()
    compress_mutually_exclusiv.add_argument(
        "-s",
        "--svd",
        action="store_true",
        help="compress an image using svd",
        default=False,
    )
    compress_mutually_exclusiv.add_argument(
        "-p",
        "--pca",
        action="store_true",
        help="compress an image using pca",
        default=False,
    )
    # pca_compressor.add_argument("image_path", type=str, help="the file path to the image you want to compress")
    # pca_compressor.add_argument("compression_level", type=int, help="Choose a compression level between 1 and 100 for your image", default=0)

    args = parser.parse_args()
    print(args.commands)
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
                        "Please indicate whether you want to convert to a jpg with -j(or --jpg) or to a png with -p(or --png)",
                        style="danger",
                    )
            else:
                source = args.file_paths[0]
                if args.jpg:
                    convert_to_jpg(source)
                elif args.png:
                    convert_to_png(source)
                else:
                    console.print(
                        "Please indicate whether you want to convert to a jpg with -j(or --jpg) or to a png with -p(or --png)",
                        style="danger",
                    )
        else:
            console.print("The number of arguments given was too many", style="danger")
    elif args.commands == "gather":
        gather(args.file_path)
    elif args.commands == "battery":
        battery()
    elif args.commands == "dark":
        toggle_dark()
    elif args.commands == "empty":
        empty_trash()
    elif args.commands == "scrape":
        scraper.scrape()
    elif args.commands == "compress":
        if args.svd:
            compress.svd_compress(args.image_path, args.compression_level)
        elif args.pca:
            compress.pca_compress(args.image_path, args.compression_level)
    # elif args.commands == ""


if __name__ == "__main__":
    """
    The main function of the CLI,
    runs the argParse
    """
    personal()
