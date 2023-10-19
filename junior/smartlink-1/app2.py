import numpy as np
import uvicorn
from fastapi import FastAPI

CLICKS_TO_OFFERS = {}
OFFER_STATS = {}

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global CLICKS_TO_OFFERS
    global OFFER_STATS
    CLICKS_TO_OFFERS = {}
    OFFER_STATS = {}


def get_offer_stats(offer_id):
    stats = OFFER_STATS.get(offer_id)

    if not stats:
        stats = {
            "clicks": 0,
            "conversions": 0,
            "reward": 0,
        }
    return stats


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    stats = get_offer_stats(offer_id)

    response = {
        "offer_id": offer_id,
        "clicks": stats["clicks"],
        "conversions": stats["conversions"],
        "reward": stats["reward"],
        "cr": stats["conversions"] / stats["clicks"] if stats["clicks"] else 0,
        "rpc": stats["reward"] / stats["clicks"] if stats["clicks"] else 0,
    }
    return response


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)
    offer_id = CLICKS_TO_OFFERS[click_id]

    if not offer_id in OFFER_STATS:
        OFFER_STATS[offer_id] = get_offer_stats(offer_id)

    is_conversion = reward != 0

    OFFER_STATS[offer_id]["conversions"] += int(is_conversion)
    OFFER_STATS[offer_id]["reward"] += reward

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward,
    }
    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    EPSILON = 0.1

    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    dice_roll = np.random.random()
    if dice_roll <= EPSILON:
        sampler = "random"
    else:
        sampler = "greedy"

    if sampler == "random":
        offer_id = int(np.random.choice(offers_ids))
    else:
        stats = [get_offer_stats(candidate) for candidate in offers_ids]

        rpcs = np.array(
            [
                (stat["reward"] / stat["clicks"] if stat["clicks"] else 0)
                for stat in stats
            ]
        )

        if rpcs.sum() == 0:
            offer_id = int(offers_ids[0])
        else:
            index_max = np.argmax(rpcs)
            offer_id = int(offers_ids[index_max])

    CLICKS_TO_OFFERS[click_id] = offer_id

    if not offer_id in OFFER_STATS:
        OFFER_STATS[offer_id] = get_offer_stats(offer_id)
    OFFER_STATS[offer_id]["clicks"] += 1

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": sampler,
    }
    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()
