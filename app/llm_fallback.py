"""LLM fallback for low-confidence transactions.

Uses the Anthropic API when ANTHROPIC_API_KEY is set; otherwise a
deterministic stub so the service runs locally without credentials.
"""
import os

CATEGORIES = [
    "groceries", "dining", "transport", "subscriptions", "utilities",
    "shopping", "travel", "health", "entertainment", "income", "transfer",
    "other",
]

_PROMPT = (
    "Categorize this bank transaction description into exactly one of: "
    f"{', '.join(CATEGORIES)}. Reply with only the category word.\n\n"
    "Transaction: {description}"
)


async def llm_categorize(description: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _stub(description)

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user",
                   "content": _PROMPT.format(description=description)}],
    )
    label = msg.content[0].text.strip().lower()
    return label if label in CATEGORIES else "other"


def _stub(description: str) -> str:
    """Keyword heuristic standing in for the LLM in local dev."""
    d = description.lower()
    table = {
        "groceries": ["whole foods", "trader joe", "kroger", "safeway", "aldi"],
        "dining": ["doordash", "grubhub", "chipotle", "starbucks", "restaurant"],
        "transport": ["uber", "lyft", "shell", "chevron", "parking", "metro"],
        "subscriptions": ["netflix", "spotify", "hulu", "openai", "anthropic"],
        "travel": ["airline", "delta", "united", "airbnb", "marriott", "hotel"],
        "utilities": ["electric", "water", "comcast", "verizon", "t-mobile"],
        "shopping": ["amazon", "amzn", "target", "walmart", "best buy"],
    }
    for cat, kws in table.items():
        if any(k in d for k in kws):
            return cat
    return "other"
