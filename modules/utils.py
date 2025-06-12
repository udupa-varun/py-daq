from datetime import datetime, timezone


def get_utcnow() -> datetime:
    utcnow = datetime.now(tz=timezone.utc)

    return utcnow


def get_utcnow_str(format_str: str = "%Y%m%d-%H%M%S") -> str:
    utcnow = get_utcnow()
    utcnow_str = utcnow.strftime(format_str)

    return utcnow_str
