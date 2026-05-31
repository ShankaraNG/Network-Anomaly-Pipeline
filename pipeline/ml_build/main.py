from ml_build.services.pipelinerunner import pipelinerunner
from ml_build.logger import get_logger
import sys

log = get_logger("Main")

def main():
    try:
        log.info("Starting the Ml build pipeline")
        results = pipelinerunner()
        if results is None or results != "successfull":
            raise Exception("Pipline did not run as expected")
        log.info("Ml build pipeline has been completed")
    except Exception as e:
        log.error("The Ml Build Pipeline has failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
