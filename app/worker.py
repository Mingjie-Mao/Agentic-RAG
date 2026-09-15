import logging
import signal
import time

from app.ingestion import claim_job, process_job

running = True


def stop(*_):
    global running
    running = False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        try:
            claim = claim_job()
            if claim:
                try:
                    process_job(*claim)
                    logging.info("ingestion completed job=%s", claim[0])
                except Exception as exc:
                    logging.warning("ingestion failed job=%s kind=%s", claim[0], type(exc).__name__)
            else:
                time.sleep(1)
        except Exception as exc:
            logging.error("worker dependency unavailable kind=%s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
