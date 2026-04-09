"""
Weather data integration using the Open-Meteo API.

Fetches hourly and daily weather data for correlating bird activity
with environmental conditions. Free API, no key required.

API docs: https://open-meteo.com/en/docs
"""

import logging
from datetime import datetime, date

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherService:
    """Fetches weather data from Open-Meteo for the configured location."""

    def __init__(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        timezone: str | None = None,
    ):
        self.latitude = latitude or settings.latitude
        self.longitude = longitude or settings.longitude
        self.timezone = timezone or settings.timezone
        self._client = httpx.Client(timeout=30.0)

    def get_current_weather(self) -> dict | None:
        """
        Fetch current weather conditions.

        Returns dict with temperature_c, humidity_pct, wind_speed_kmh,
        precipitation_mm, cloud_cover_pct, weather_code, or None on failure.
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
                "weather_code",
            ],
            "timezone": self.timezone,
        }

        logger.debug(f"Fetching current weather for ({self.latitude}, {self.longitude})")
        try:
            response = self._client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            result = {
                "temperature_c": current.get("temperature_2m"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "precipitation_mm": current.get("precipitation"),
                "cloud_cover_pct": current.get("cloud_cover"),
                "weather_code": current.get("weather_code"),
                "timestamp": datetime.fromisoformat(current["time"]) if "time" in current else datetime.now(),
            }
            logger.debug(
                f"Weather: {result['temperature_c']}C, "
                f"{result['humidity_pct']}% humidity, "
                f"code={result['weather_code']}"
            )
            return result
        except (httpx.HTTPError, KeyError) as e:
            logger.error(f"Failed to fetch current weather: {e}")
            return None

    def get_daily_sun_times(self, target_date: date | None = None) -> dict | None:
        """
        Fetch sunrise and sunset times for a given date.

        Returns dict with sunrise and sunset as datetime objects, or None on failure.
        """
        target_date = target_date or date.today()

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": ["sunrise", "sunset"],
            "timezone": self.timezone,
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
        }

        logger.debug(f"Fetching sun times for {target_date}")
        try:
            response = self._client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            sunrise_list = daily.get("sunrise", [])
            sunset_list = daily.get("sunset", [])

            if sunrise_list and sunset_list:
                result = {
                    "sunrise": datetime.fromisoformat(sunrise_list[0]),
                    "sunset": datetime.fromisoformat(sunset_list[0]),
                }
                logger.debug(
                    f"Sun times: sunrise={result['sunrise'].strftime('%H:%M')}, "
                    f"sunset={result['sunset'].strftime('%H:%M')}"
                )
                return result

            logger.error(f"Open-Meteo returned empty sun times for {target_date}")
        except (httpx.HTTPError, KeyError, IndexError) as e:
            logger.error(f"Failed to fetch sun times: {e}")

        return None

    def get_hourly_weather(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """
        Fetch hourly weather data for a date range.

        Defaults to today. Returns a list of hourly observation dicts.
        """
        start_date = start_date or date.today()
        end_date = end_date or start_date

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
                "weather_code",
            ],
            "timezone": self.timezone,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        try:
            response = self._client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            observations = []
            for i, time_str in enumerate(times):
                observations.append({
                    "timestamp": datetime.fromisoformat(time_str),
                    "temperature_c": hourly.get("temperature_2m", [None])[i],
                    "humidity_pct": hourly.get("relative_humidity_2m", [None])[i],
                    "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[i],
                    "precipitation_mm": hourly.get("precipitation", [None])[i],
                    "cloud_cover_pct": hourly.get("cloud_cover", [None])[i],
                    "weather_code": hourly.get("weather_code", [None])[i],
                })

            return observations
        except (httpx.HTTPError, KeyError) as e:
            logger.error(f"Failed to fetch hourly weather: {e}")
            return []

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Weather code descriptions from Open-Meteo WMO codes
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def describe_weather_code(code: int | None) -> str:
    """Convert a WMO weather code to a human-readable description."""
    if code is None:
        return "Unknown"
    return WEATHER_CODES.get(code, f"Unknown ({code})")
