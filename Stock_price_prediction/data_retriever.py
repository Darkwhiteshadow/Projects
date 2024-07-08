import requests
import pandas as pd 

key ="497b8f719eca7942be1ea0fd731e88517396bdfb"
# Defineing the endpoint and parameters
ticker = "MSFT"  # Fetching data for Microsoft (MSFT)
url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
params = {
    'token': key,
    'startDate': '2015-01-01',
    'endDate': '2024-05-10'
}

response =requests.get(url,params=params)

if response.status_code == 200:

    data  = response.json()

    df = pd.DataFrame(data)

    df.to_csv('MSTF.csv',index=False)

    print("Datasave as MSTF.csv")

else:

    print(f'Failed to fetch data: {response.status_code}')

