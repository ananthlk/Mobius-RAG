"""Allow ``python -m app.worker`` to launch the chunking worker."""
from app.worker.main import main

if __name__ == "__main__":
    main()
