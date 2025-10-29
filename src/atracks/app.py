import argparse
import logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
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
        "-s",
        "--spindle",
        dest="spindle_channel",
        type=int,
        help="channel # for tracking spindle poles",
    )
    parser.add_argument(
        "-d",
        "--dna",
        dest="dna_channel",
        type=int,
        help="channel # for tracking dna",
    )
    parser.add_argument(
        "-r",
        "--refframe",
        dest="reference_frame",
        type=int,
        default=0,
        help=(
            "reference frame to determine spindle pole axis (0=autodetect based on cell long axis)"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging as early as possible
    level_name = str(args.loglevel).upper() if getattr(args, "loglevel", None) else "INFO"
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("atracks invoked with args: %s", vars(args))
    parser.print_help()


if __name__ == "__main__":
    main()

