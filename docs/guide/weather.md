# Weather in Live

Enable **Live weather tile** under **Admin → Server Preferences → General**,
enter a latitude and longitude and location name, then choose **Save settings**.
The location is shared by all viewers. Weather is disabled by default and does
not require an API key. Changing these settings does not restart cameras.

The tile joins the Live grid, including automatic layout, custom move/resize,
desktop focus and mobile primary selection. Its expand button opens a large
radar map with zoom controls and the next twelve hourly forecast entries.
Escape or Close returns to the grid. Weather has no recording, detection,
ONVIF, or camera-health controls and does not consume a camera worker.

The compact view shows temperature, feels-like temperature, humidity, wind and
gusts. Its recent radar loop can be paused or scrubbed. Animation pauses when
the tile or browser tab is hidden, and starts paused when the browser requests
reduced motion. Temperatures and wind speeds support °F/mph or °C/km/h.

## Providers and freshness

- [Open-Meteo](https://open-meteo.com/en/docs) supplies model-derived current
  conditions and hourly forecasts, rather than a measurement at the property.
  SurvNG caches successful conditions for ten minutes.
- [RainViewer](https://www.rainviewer.com/api/weather-maps-api.html) supplies
  recent radar frames and the coverage mask. The documented history is two
  hours at ten-minute intervals, with zoom limited to level 7. SurvNG caches
  radar metadata for five minutes. The displayed radar time is the composite
  frame generation time; individual radar observations can be older. This
  version does not display future radar predictions or severe-weather alerts.
- [OpenStreetMap](https://www.openstreetmap.org/copyright) supplies the basemap.
  Map and radar imagery load directly in the browser, using normal HTTP caching
  and visible attribution. Shaded regions indicate missing radar coverage;
  an image-load error is displayed separately from clear weather.

The browser polls visible weather once per minute; upstream requests are shared
across viewers. Failures back off from thirty seconds to fifteen minutes and
retain the last successful response. Conditions and radar show stale status
independently. A changed location or units clears the conditions cache so a
failed request cannot show the previous location's readings. Loading, missing
data and failed imagery have explicit states.

Conditions requests send the coordinates to Open-Meteo from the server.
Browser map requests reveal the selected region to the imagery providers.
Read the [Open-Meteo usage terms](https://open-meteo.com/en/pricing),
[RainViewer terms](https://www.rainviewer.com/api.html), and
[OSM tile policy](https://operations.osmfoundation.org/policies/tiles/) before
enabling the feature. The free weather services target personal/noncommercial
use and provide no availability guarantee. The v1 providers are fixed in code.

## Configuration

The optional top-level configuration is:

```json
"weather": {
  "enabled": false,
  "name": "Local weather",
  "latitude": null,
  "longitude": null,
  "units": "imperial",
  "radar_zoom": 6,
  "animate": true
}
```

Enabling weather requires both coordinates. Latitude is limited to −85…85
for the map projection; longitude to −180…180, and zoom to 2…7.
The read-scoped `/api/weather/conditions` and `/api/weather/radar` endpoints
use the existing SurvNG API authentication. They return 404 while disabled.
