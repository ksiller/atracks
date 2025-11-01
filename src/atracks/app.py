import argparse
import logging

from .utils import load_mp4, check_device
from .analysis import analyze


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser for the atracks CLI.
    """
    parser = argparse.ArgumentParser(
        prog="atracks",
        description=(
            "Tool to track atom coordination over time."
        ),
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        dest="loglevel",
        type=str,
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        help="single image file or directory with image files to be processed",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="output file or directory",
    )
    parser.add_argument(
        "-g",
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="convert loaded videos to grayscale before processing",
    )
    parser.add_argument(
        "-f",
        "--frames",
        dest="frames",
        type=str,
        default=None,
        help="frames to process; examples: '10' or '2-40' (inclusive). Default: all frames",
    )
    return parser


def main() -> None:
    """Entry point for the atracks CLI.

    Parses command-line arguments, configures logging, and invokes analysis.

    Returns:
        None
    """
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging as early as possible
    level_name = str(args.loglevel).upper() if getattr(args, "loglevel", None) else "INFO"
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Determine compute device
    device = check_device()
    logger.info("Selected device: %s", device)

    logger.info("atracks invoked with args: %s", vars(args))

    # Parse frames into start/end integers
    start_frame = None
    end_frame = None
    if args.frames:
        fs = str(args.frames).strip()
        if "-" in fs:
            a, b = fs.split("-", 1)
            if not a.isdigit() or not b.isdigit():
                raise ValueError("--frames must be positive integers like '10' or '2-40'")
            a_i = int(a)
            b_i = int(b)
            if a_i < 1 or b_i < 1:
                raise ValueError("--frames indices are 1-based and must be >= 1")
            if b_i < a_i:
                raise ValueError("--frames end must be >= start")
            # Convert to zero-based inclusive indices
            start_frame = a_i - 1
            end_frame = b_i - 1
        else:
            if not fs.isdigit():
                raise ValueError("--frames must be a positive integer or a range 'A-B'")
            n = int(fs)
            if n < 1:
                raise ValueError("--frames index is 1-based and must be >= 1")
            start_frame = n - 1
            end_frame = start_frame
    
    logger.info("Opening: %s", input)
    image_stack = load_mp4(Path(input), to_grayscale=args.grayscale, start_frame=start_frame, end_frame=end_frame)
    logger.info("Image stack: %s", image_stack.shape)

    results =analyze(
        input=image_stack)
    logger.info("Results: %s", results) 



if __name__ == "__main__":
    main()

