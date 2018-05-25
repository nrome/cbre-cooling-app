# Datacenter Cooling Unit Optimization

This repo houses the source code for the Datacenter Cooling Unit Optimization Web Application which is repsonsible for visualizing:
 - the output of the optimization model simulation on the historical data
 - the application of the operational issues detection algorithms on the historical data

## Architecture
This application's architecture is very simple in terms of components:
 - A single python-flask application that has the front and back end code in one place
 - A db2 warehouse on cloud database that has all of the data

Aside from the web service, this web application also has exposed RESTful API's which visualize the analytics (heatmap, operational issues, the operational health dashboard, among other things). Examples of the methods are at the bottom of this README.

## Next Steps
The critical next steps in the development of this application are the following:
1. Create the load data screen which is going to populate the database based on static spreadsheets or csv files loaded (so that this is able to perform its function as a screening tool)
2. Update some of the hardcoded elements of the code (i.e. the CRAH unit codes) so that this becomes purely dynamic (right now it's at ~90% dynamic)
3. Integrate IBM data lift into the architecture to make it seamless to onboard new data

## API Methods
1. Analytics
method: analytics, POST
### Payload
```
payload = {
    "equipment": anyOf(equipment_list),
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "plot_format": "html" or "json" (html if html string else json if object)
}
```

```
equipment_list = {
            "DH1", "DH2", "DH4", "DH5", "DH9", "DH10", "DH11", "DH14", "DH15",
            "DH17", "DH18", "DH21", "DH23", "DH24", "DH26", "OAT", "DP"
}
```
where DH1 to DH26 refer to crah units, OAT refers to the outside air
temperature sensor, and DP refers to the differential pressure sensor

### Example
```python
import requests
payload = {
    "equipment": "DP",
    "start": "2017-10-01",
    "end": "2017-11-01",
    "plot_format": "html"
}
url = 'http://viz-engine.mybluemix.net'
response = requests.post(url + '/api/v1/analytics/', json=payload)
```

### Response
```
response = {
    "plots": [
        {
            "plot": html string or json dict of plots,
            "plot_name": string name of plot,
            "suggested_action": string name of suggested action

    ]
}
```

#### NOTE
Eventually, there is going to be another argument in this payload
named f_periods: where f_periods refers to how many forecast periods
there are going to be such that the payload will be as below, and the plot
will have both the historical data from start to end plus a forecasted
number of f_periods periods

```
payload = {
    "equipment": anyOf(equipment_list),
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "plot_format": "html" or "json" (html if html string else json if object)
    "f_periods": int
}
```
example
```python
payload = {
    "equipment": "DP",
    "start": "2017-10-01",
    "end": "2017-11-01",
    "plot_format": "html"
    "f_periods": 2016
}
# a single period in our data is 5 minutes. To forecast one day means 288
# periods (24 hours * 12 periods/hour), one week is 2016
# (24 hours * 7 days * 12 periods/hour)
```

2. Dashboard
method: dashboard, POST
### Payload
```
payload = {
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "plot_format": "html" or "json" (html if html string else json if object)
}
```

### Example
```python
import requests
payload = {
    "start": "2017-10-01",
    "end": "YYYY-MM-DD",
    "plot_format": "html",
}
url = 'http://viz-engine.mybluemix.net'
response = requests.post(url + '/api/v1/dashboard/', json=payload)
```

### Response
```
response = {
    "plots": [
        {
            "plot": html string or json dict of plots,
            "plot_name": string name of plot,
        }
    ]
}
```

3. Heat Map
method: heat_map, POST
### Payload
Only the start time is necessary because based on this, a heatmap will
be produced as well as the forecasted heatmaps
```
payload = {
    "start": "YYYY-MM-DD hh:mm:ss",
    "plot_format": "html" or "json" (html if html string else json if object)
    "forecasted": "YYYY-MM-DD hh:00:00" this is the forecasted time period
    ahead of the current start period (since a baseline of now needs to be
    detremined in creating the forecasts)
}
```

### Example
```python
import requests
payload = {
    "start": "2017-05-05 00:00:00",
    "plot_format": "html",
    "forecasted": "2017-10-01 01:00:00"
}
url = 'http://viz-engine.mybluemix.net'
response = requests.post(url + '/api/v1/heat_map/', json=payload)
```

### Response
This response is automatically going to contain XX forecasted heatmaps
(XX could be 12 or 24 at the moment), the plot_names will be "NOW",
"NOW + 1 HOUR(S)", ... "NOW + 12 HOUR(S)"
```
response = {
    "plots": [
        {
            "plot": html string or json dict of plots,
            "plot_name": string name of plot,
        }
    ]
}
```
