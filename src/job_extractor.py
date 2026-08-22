import requests
from bs4 import BeautifulSoup


def extract_job_from_url(url):
    """
    Extract readable text from a public job-page URL.

    Some websites, including platforms with anti-bot
    protection, may block automated requests.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
    ]):
        element.decompose()

    text = soup.get_text(
        separator="\n"
    )

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    cleaned_text = "\n".join(lines)

    if len(cleaned_text) < 300:

        raise ValueError(
            "The webpage did not provide enough readable "
            "job-description text. The website may be "
            "blocking automated extraction."
        )

    return cleaned_text