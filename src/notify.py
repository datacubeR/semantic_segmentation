import requests


def notify(message: str, title: str = "Training", priority: str = "3"):
    requests.post(
        "https://ntfy.sh/semantic-segmentation",
        data=message.encode("utf-8"),
        headers={
            "Title": title,
            "Priority": priority,
        },
    )
