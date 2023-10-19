import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/sample/")
def sample(offer_ids: str) -> dict:
    """Sample random offer"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Sample random offer ID
    # offer_id = sorted(offers_ids)[0]
    offer_id = int(np.random.choice(offers_ids))

    # Prepare response
    response = {
        "offer_id": offer_id,
    }

    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()
