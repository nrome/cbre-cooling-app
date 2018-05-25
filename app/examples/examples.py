import requests
import pandas as pd
import timeit

url = 'http://viz-engine.mybluemix.net'


def test_query():
    start = timeit.default_timer()
    query = '''SELECT *
               FROM DASH5526.DH4
               WHERE "date_time" BETWEEN '2017-04-01' AND '2017-04-07'
               ORDER BY "date_time" ASC;'''
    payload = {
        "query": query
    }
    response = requests.post(url + '/api/v1/query/', json=payload)
    df = pd.DataFrame(response.json()['results'])
    df.set_index('date_time', inplace=True)
    df = df.astype(float)
    print df.head()
    if response.status_code == 200:
        print 'db connection successful'
    else:
        print response.content
    print 'query took', timeit.default_timer() - start


def test_analytics():
    start = timeit.default_timer()
    payload = {
        "equipment": "DP",
        "start": "2017-10-01",
        "end": "2017-10-07",
        "plot_format": "html"
    }
    response = requests.post(url + '/api/v1/analytics/', json=payload)
    if response.status_code == 200:
        print 'differential pressure issues successful'
    else:
        print response.content
    print 'dp plot took', timeit.default_timer() - start

    start = timeit.default_timer()
    payload = {
        "equipment": "DH1",
        "start": "2017-10-01",
        "end": "2017-10-07",
        "plot_format": "html"
    }
    response = requests.post(url + '/api/v1/analytics/', json=payload)
    if response.status_code == 200:
        print 'crah unit issues successful'
    else:
        print response.content
    print 'crah plot took', timeit.default_timer() - start

    start = timeit.default_timer()
    payload = {
        "equipment": "OAT",
        "start": "2017-10-01",
        "end": "2017-10-07",
        "plot_format": "html"
    }
    response = requests.post(url + '/api/v1/analytics/', json=payload)
    if response.status_code == 200:
        print 'oat sensor issues successful'
    else:
        print response.content
    print 'oat plot took', timeit.default_timer() - start


def test_dashboard():
    start = timeit.default_timer()
    payload = {
        "start": "2017-10-01",
        "end": "2017-10-07",
        "plot_format": "html"
    }
    response = requests.post(url + '/api/v1/dashboard/', json=payload)
    if response.status_code == 200:
        print 'dashboard successful'
    else:
        print response.content
    print 'dashboard took', timeit.default_timer() - start


if __name__ == "__main__":
    test_query()
    test_analytics()
    test_dashboard()
