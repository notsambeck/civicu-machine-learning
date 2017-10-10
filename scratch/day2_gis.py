import pandas as pd
import re
import requests
import time

df = pd.read_csv('lessons/shared-resources/crimedata.csv', index_col='id')

# del(df['Unnamed: 0'])

print(df.head())

h = '3025+NE+Hoyt+St.,+Portland,+OR'

def geocode_osm(address, polygon=0):
    polygon = int(polygon)
    address = address.replace(' ', '+').replace('\r\n',',').replace('\r',',').replace('\n',',')
    osm_url = 'http://nominatim.openstreetmap.org/search'
    osm_url += '?q={address}&format=json&polygon={polygon}&addressdetails={addressdetails}'.format(
        address=address, polygon=polygon, addressdetails=0)

    print(osm_url)
    resp = requests.get(osm_url)
    print(resp)
    d = resp.json()
    print(d)

    return {
        'lat': d[0].get('lat', pd.np.nan),
        'lon': d[0].get('lon', pd.np.nan),
       }


def geocode_google(address, apikey='AIzaSyAyUtgR634vpwNkpfewBY0nVcWiWqgs3-Y'):
    # https://developers.google.com/maps/documentation/embed/get-api-key
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(address, apikey)
    print(url)
    resp = requests.get(url)
    results = resp.json()
    results = results.get('results', {})
    results = [{}] if not len(results) else results
    latlon = results[0].get('geometry', {}).get('location', {})
    return {
            'lat': latlon.get('lat', pd.np.nan),
            'lon': latlon.get('lng', pd.np.nan),
           }


def simplify_address(address, remove_zip=True, remove_apt=True):
    address = address.lower().replace(' ', '+')
    zipcode = re.compile('[0-9]{4,5}[-]?[0-9]{0,4}$')
    address = zipcode.sub('', address or '')
# aptnum = re.compile('(\b#[ ]?|apt|unit|appartment)\s?([A-Z]?[-]?[0-9]{0,6})')
# address = aptnum.sub('', address or '')
    return address


def get_lat_long(crime_iter):
    return geocode_google(simplify_address(next(crime_iter).address))


def get_first_ten_coords():
    crime_iter = df.itertuples()
    out = []
    for i in range(10):
        try:
            loc = (get_lat_long(crime_iter))
        except:
            print('api call failed to produce coords')
            loc = 'fail'
        out.append(loc)
        print(loc)
        time.sleep(1)
    return out

'''
from sklearn.sometihng import LabelEncoder
# label encoder - good for ordinals; not good for neighborhoods
le = LabelEncoder()
le.fit(df.neighborhood)
df['neighborhood_int'] = le.transform(df.neighborhood)
'''

neighborhood = pd.get_dummies(df.neighborhood)
neighborhood.head(3)

# select burglaries
crime = pd.get_dummies(df.neighborhood)
bulgraries = df.df['major_offense_type'] == 'Burglary'
# or alternatively
df[crime.Burlglary.astype(bool)]
