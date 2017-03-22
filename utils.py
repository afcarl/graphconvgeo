'''
Created on 5 Mar 2017

@author: af
'''
import numpy as np
#from matplotlib.patches import Polygon
import pdb
import json
import pickle
from collections import Counter

short_state_names = {
       # 'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
       # 'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        #'GU': 'Guam',
       # 'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
      #  'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        #'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

stop_words = ['the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'all', 'have', 'new', 'more', 'an', 'was', 'we', 'will', 'home', 'can', 'us', 'about', 'if', 'page', 'my', 'has', 'search', 'free', 'but', 'our', 'one', 'other', 'do', 'no', 'information', 'time', 'they', 'site', 'he', 'up', 'may', 'what', 'which', 'their', 'news', 'out', 'use', 'any', 'there', 'see', 'only', 'so', 'his', 'when', 'contact', 'here', 'business', 'who', 'web', 'also', 'now', 'help', 'get', 'pm', 'view', 'online', 'c', 'e', 'first', 'am', 'been', 'would', 'how', 'were', 'me', 's', 'services', 'some', 'these', 'click', 'its', 'like', 'service', 'x', 'than', 'find', 'price', 'date', 'back', 'top', 'people', 'had', 'list', 'name', 'just', 'over', 'state', 'year', 'day', 'into', 'email', 'two', 'health', 'n', 'world', 're', 'next', 'used', 'go', 'b', 'work', 'last', 'most', 'products', 'music', 'buy', 'data', 'make', 'them', 'should', 'product', 'system', 'post', 'her', 'city', 't', 'add', 'policy', 'number', 'such', 'please', 'available', 'copyright', 'support', 'message', 'after', 'best', 'software', 'then', 'jan', 'good', 'video', 'well', 'd', 'where', 'info', 'rights', 'public', 'books', 'high', 'school', 'through', 'm', 'each', 'links', 'she', 'review', 'years', 'order', 'very', 'privacy', 'book', 'items', 'company', 'r', 'read', 'group', 'sex', 'need', 'many', 'user', 'said', 'de', 'does', 'set', 'under', 'general', 'research', 'university', 'january', 'mail', 'full', 'map', 'reviews', 'program', 'life']

def get_us_city_name():
    #we might exclude words in city names
    all_us_city_names = set()
    with open('./data/us_cities.txt', 'r') as fin:
        for line in fin:
            words = set(line.strip().lower().split())
            for word in words:
                all_us_city_names.add(word)
    return all_us_city_names

def retrieve_location_from_coordinates():

    points = []
    #read points from a file
    with open('./data/latlon_world.txt', 'r') as fin:
        for line in fin:
            line = line.strip()
            lat, lon = line.split('\t')
            lat, lon = float(lat), float(lon)
            point = (lat, lon)
            points.append(point)
    #read point city-countries from http://people.eng.unimelb.edu.au/tbaldwin/resources/jair2014-geoloc/
    latlon_country = {}
    with open('./data/han_cook_baldwin.geo', 'r') as fin:
        for line in fin:
            line = line.strip()
            fields = line.split('\t')
            country = fields[0].split('-')[-1].upper()
            lat = float(fields[2])
            lon = float(fields[3])
            latlon_country[(lat, lon)] = country
    
    country_count = Counter()
    for point in points:
        country = latlon_country[point]
        country_count[country] += 1
    countries = [c for c, count in country_count.iteritems() if count>100]
    with open('./data/country_count.txt', 'w') as fout:
        json.dump(countries, fout)

if __name__ == '__main__':
    retrieve_location_from_coordinates()
        
        
    
        
