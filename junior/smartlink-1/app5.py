import numpy as np
import uvicorn
from fastapi import FastAPI

CLICKS_TO_OFFERS = {}
DISTRIBUTIONS = {}
OFFER_STATS = {}

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global CLICKS_TO_OFFERS
    global OFFER_STATS
    global DISTRIBUTIONS
    CLICKS_TO_OFFERS = {}
    OFFER_STATS = {}
    DISTRIBUTIONS = {}


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

    if not offer_id in DISTRIBUTIONS:
        DISTRIBUTIONS[offer_id] = {
            "alpha": 1,
            "beta": 1,
        }

    is_conversion = reward != 0

    if is_conversion:
        DISTRIBUTIONS[offer_id]["alpha"] += 1
    else:
        DISTRIBUTIONS[offer_id]["beta"] += 1

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
    offer_ids = [int(offer) for offer in offer_ids.split(",")]

    selected_offer_id = None
    samples = []
    for offer_id in offer_ids:
        if not offer_id in OFFER_STATS:
            OFFER_STATS[offer_id] = get_offer_stats(offer_id)

        if not offer_id in DISTRIBUTIONS:
            DISTRIBUTIONS[offer_id] = {
                "alpha": 1,
                "beta": 1,
            }
        alpha = DISTRIBUTIONS[offer_id]["alpha"]
        beta = DISTRIBUTIONS[offer_id]["beta"]
        sample_conversion = np.random.beta(alpha, beta)
        sample_reward = OFFER_STATS[offer_id]["reward"] * sample_conversion
        samples.append(sample_reward)

    max_idx = np.argmax(samples)
    selected_offer_id = offer_ids[max_idx]

    CLICKS_TO_OFFERS[click_id] = selected_offer_id

    OFFER_STATS[selected_offer_id]["clicks"] += 1

    response = {
        "click_id": click_id,
        "offer_id": selected_offer_id,
        # "sampler": "thompson",
    }
    return response


def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost")


if __name__ == "__main__":
    main()
