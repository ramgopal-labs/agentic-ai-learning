import requests

from app.core.config import settings


def search_authoritative_sources(question: str, limit: int = 5) -> list[dict]:
    """Search only Indian legal sources when the local index is not confident."""
    if not settings.serper_api_key:
        return []

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json",
        },
        json={
            "q": f"{question} site:indiacode.nic.in OR site:indiankanoon.org",
            "num": limit,
        },
        timeout=15,
    )
    response.raise_for_status()

    return [
        {
            "title": item.get("title", "Untitled source"),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        }
        for item in response.json().get("organic", [])
    ]
