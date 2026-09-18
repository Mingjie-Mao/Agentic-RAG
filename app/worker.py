import logging
import signal
import time

from app.ingestion import claim_job, process_job
from agent.controller import claim_agent_task, process_agent_task

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
            ingestion_claim = claim_job()
            if ingestion_claim:
                try:
                    process_job(*ingestion_claim)
                    logging.info("ingestion completed job=%s", ingestion_claim[0])
                except Exception as exc:
                    logging.warning(
                        "ingestion failed job=%s kind=%s",
                        ingestion_claim[0],
                        type(exc).__name__,
                    )
            agent_claim = claim_agent_task()
            if agent_claim:
                try:
                    process_agent_task(*agent_claim)
                    logging.info("agent completed task=%s", agent_claim[0])
                except Exception as exc:
                    logging.warning(
                        "agent failed task=%s kind=%s", agent_claim[0], type(exc).__name__
                    )
            if not ingestion_claim and not agent_claim:
                time.sleep(1)
        except Exception as exc:
            logging.error("worker dependency unavailable kind=%s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
