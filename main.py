from datetime import datetime, timezone
from pathlib import Path
import argparse
import asyncio
import json
import logging

from modules.datacollector import DataCollector
from modules.utils import get_utcnow_str


def setup_logging(log_id):
    # create logs dir if it doesn't already exist
    logs_dir = Path(__file__).parent.resolve() / "logs"
    logs_dir.mkdir(exist_ok=True)

    # logging configuration
    logger = logging.getLogger("pdx")
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    formatter = logging.Formatter(
        "%(asctime)s | [%(levelname)s]:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(f"./logs/{log_id}.log", encoding="utf-8", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def validate_config(config):
    """Validate and set defaults for config file"""
    required_fields = [
        # "endpoint",
        "asset",
        "out_data_dir",
        # "interval",
        "duration",
        # "node_info",
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Set defaults for optional fields
    if "nidaq" not in config:
        config["nidaq"] = {"device": "Dev1", "channels": {}, "sample_rate": 25600}

    if "trigger" not in config:
        # If no trigger specified, use an always-true condition
        config["trigger"] = {
            "start": {"type": "opcua", "conditions": [], "logic": "and"}
        }

    return config


async def main(args):
    # Load and validate configuration
    try:
        with open(Path(args["config"]), "r") as f:
            config = json.load(f)
        config = validate_config(config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Setup logging
    fmt = "%Y%m%d-%H%M%S"
    utcnow_str = get_utcnow_str(format_str=fmt)
    log_id = f"{config['asset']}_{utcnow_str}"
    logger = setup_logging(log_id)
    logger.info("Starting data collection")

    # Prepare output file path
    output_file_name = f"{config['asset']}-{utcnow_str}.h5"
    output_file_path = Path(config["out_data_dir"]).resolve() / output_file_name

    # Create and run data collector
    try:
        async with DataCollector(config) as collector:
            logger.info("Waiting for trigger conditions...")
            success = await collector.collect_data(output_file_path)

            if success:
                logger.info("Data collection completed successfully")
                logger.info(f"Data written to {output_file_path}")
            else:
                logger.error("Data collection failed")
    except Exception as e:
        logger.error(f"Error during data collection: {e}")


if __name__ == "__main__":
    start_time = datetime.now(tz=timezone.utc)
    fmt = "%Y%m%d-%H%M%S"
    start_time_formatted = start_time.strftime(fmt)

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument(
        "-c", "--config", type=str, help="Path to config file", default="./config.json"
    )

    args = vars(ap.parse_args())

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
