import flask
from flask import Flask,redirect
app = flask.Flask(__name__)


#-------- Pre-production -----------#
import pandas as pd
import numpy as np
import json
import calendar
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import cluster, preprocessing, metrics
from sklearn.manifold import TSNE
from collections import Counter

from datetime import datetime
import calendar
import string
import time
import hashlib

from IPython.display import IFrame
from IPython.core.display import display


class geoloc(object):

    def __init__(self,geolocfile):
        try:
            self.g_loc_raw = pd.read_json(geolocfile)

        except IOError:
            print "File not found or could not be read."

        gmaps_columns = ['timestamp','lat','lng','acc','vel','heading','altitude','v_acc']
        gmaps_coord_disp = 10.**7

        loc_hist = []

        for datapoint in self.g_loc_raw['locations']:

            try:
                timestamp = datetime.fromtimestamp((int(datapoint['timestampMs'])/1000)).strftime('%Y-%m-%d %H:%M:%S')
            except:
                timestamp = ''
            try:
                lat = datapoint['latitudeE7']/(gmaps_coord_disp)
            except:
                lat = '0'
            try:
                lng = datapoint['longitudeE7']/(gmaps_coord_disp)
            except:
                lng = '0'
            try:
                alt = datapoint['altitude']
            except:
                alt = '0'
            try:
                v_acc = datapoint['verticalAccuracy']
            except:
                v_acc = '0'
            try:
                vel = datapoint['velocity']
            except:
                vel = '0'
            try:
                heading = datapoint['heading']
            except:
                heading = '0'
            try:
                acc = datapoint['accuracy']
            except:
                acc = '0'

            loc_hist.append([timestamp, lat, lng, acc, vel, heading, alt, v_acc])

        self.gloc_hist = pd.DataFrame(loc_hist, columns=gmaps_columns)

    def cluster_proc(self):
        geoc = np.array(self.gloc_hist[['lat','lng']])

        kmlist = []

        for n in range(2,100):
            lkm = KMeans(n_clusters=n)
            ltkm = lkm.fit(geoc)
            ltkm_score = metrics.silhouette_score(geoc, ltkm.labels_, metric='euclidean')
            kmlist.append([n,ltkm_score])

        klist = np.array(kmlist)


        for n in klist:
            if n[1]==np.amax(klist[(klist[:,1]<0.75),1]):
                n_max =n[0]
                sc_max = np.amax(klist[(klist[:,1]<0.75),1])

#         for n in klist:
#             if n[1]==np.amax(klist[3:100,1]):
#                 n_max =n[0]
#                 sc_max = np.amax(klist[3:100,1])

        lkm = KMeans(n_clusters=int(n_max))
        ltkm = lkm.fit(geoc)

        self.gloc_hist = self.gloc_hist.drop(['vel'], axis=1).drop(['altitude'], axis=1)
        self.gloc_hist['poi_id'] = ltkm.labels_

        self.gloc_hist['lat4'] = [round(glat,4) for glat in self.gloc_hist['lat']]
        self.gloc_hist['lng4'] = [round(glng,4) for glng in self.gloc_hist['lng']]
        self.gloc_hist['lat5'] = [round(glat,5) for glat in self.gloc_hist['lat']]
        self.gloc_hist['lng5'] = [round(glng,5) for glng in self.gloc_hist['lng']]

        self.poi_id_list = self.gloc_hist['poi_id'].value_counts().to_frame(name='Count').reset_index()
        self.poi_sig = self.poi_id_list['Count'].describe()

        self.min_sig = self.poi_sig.ix['mean']+self.poi_sig.ix['std']

        if len(self.poi_id_list[(self.poi_id_list['Count']>self.min_sig)]['index'].values) < 2:
            self.min_sig = self.poi_sig.ix['mean']+self.poi_sig.ix['std']
        else:
            self.min_sig = self.poi_sig.ix['mean']+(self.poi_sig.ix['std']*2)

        self.poi_points = self.poi_id_list[(self.poi_id_list['Count']>self.min_sig)]['index'].values

        self.gloc_hist['timestamp'] = pd.to_datetime(self.gloc_hist['timestamp'])
        self.gloc_hist['dotw'] = [gldt.dayofweek for gldt in self.gloc_hist['timestamp']]
        self.gloc_hist['hotd'] = [gldt.hour for gldt in self.gloc_hist['timestamp']]

        self.poi_dotw_df_list = []

        for n_poi in range(0,len(self.poi_points)):
            self.poi_dotw_df_list.append(self.gloc_hist[(self.gloc_hist['poi_id']==self.poi_points[n_poi])]['dotw'].describe().values)

        poi_freq_cols = ['count','mean','std','min','25p','50p','75p','max']
        poi_freq_df = pd.DataFrame(self.poi_dotw_df_list,columns=poi_freq_cols, index=[self.poi_points])

        #----------
        dayfreq_list = []
        cols = ['weekday_visits','weekend_visits','weekday_only','weekend_only']

        for poi_loc in self.poi_points:

            on_weekends = []
            on_weekdays = []
            visit_weekends = []
            visit_weekdays = []
            weekdays_only = []
            weekends_only = []
            weekdays = (0,1,2,3,4)
            weekends = (5,6)

            for dotw in self.gloc_hist[(self.gloc_hist['poi_id']==poi_loc)]['dotw'].value_counts().index:
                if dotw in weekends:
                    on_weekends.append(dotw)
                if dotw in weekdays:
                    on_weekdays.append(dotw)

            visit_weekends.append(len(on_weekends)==0)
            visit_weekdays.append(len(on_weekdays)==0)
            weekdays_only.append((len(on_weekends)==0) & (len(on_weekdays)>0))
            weekends_only.append((len(on_weekdays)==0) & (len(on_weekends)>0))

            dayfreq_list.append([on_weekdays,on_weekends,weekdays_only,weekends_only])

        self.poi_dayfreq_df = pd.DataFrame(dayfreq_list, columns=cols,index=[self.poi_points])
        self.poi_dayfreq_df['weekday_only'] = [item[0] for item in self.poi_dayfreq_df['weekday_only']]
        self.poi_dayfreq_df['weekend_only'] = [item[0] for item in self.poi_dayfreq_df['weekend_only']]
        self.poi_main_df = pd.concat([poi_freq_df, self.poi_dayfreq_df], axis=1)

        self.selected_poi_for_freq_query = poi_freq_df['std'].idxmin()

        self.most_freq_dotw_for_pot = self.gloc_hist[(self.gloc_hist['poi_id']==self.selected_poi_for_freq_query)]['dotw'].value_counts().index[0]


        #--------------

        poi_freq_df = pd.DataFrame(self.poi_dotw_df_list,columns=poi_freq_cols, index=[self.poi_points])
        self.selected_poi_for_freq_query = poi_freq_df['std'].idxmin()
        self.most_freq_dotw_for_pot = self.gloc_hist[(self.gloc_hist['poi_id']==self.selected_poi_for_freq_query)]['dotw'].value_counts().index[0]

        self.valid_answer = []

    def mean_lat(self):
        return self.gloc_hist[(self.gloc_hist['poi_id']==self.selected_poi_for_freq_query)]['lat'].mean()

    def mean_lng(self):
        return self.gloc_hist[(self.gloc_hist['poi_id']==self.selected_poi_for_freq_query)]['lng'].mean()

    def get_glhist(self):
        return self.gloc_hist

    def showmap(self,maplat,maplng,zoom=16):
        maps_url = "http://maps.google.com/maps?q={0}+{1}&z={2}&output=embed&iwloc=near".format(maplat,maplng,zoom)
        display(IFrame(maps_url, '400px', '300px'))

    def getmapurl(self,maplat,maplng,zoom=16):
        maps_url = "http://maps.google.com/maps?q={0}+{1}&z={2}&output=embed&iwloc=near".format(maplat,maplng,zoom)
        return maps_url

class lbc(object):

    testdevid = '1425272220649281'

    def __init__(self,filename):
        self.token = 'a'
        self.geoprocdata = geoloc(filename)
        self.geoprocdata.cluster_proc()
        self.tries=2

    def tokemon(self,devid = testdevid):
        self.to_encode = devid + str(int(round(time.time())))
        self.encoded = hashlib.sha224(self.to_encode).hexdigest()
        self.token = self.encoded
        return self.token

#     def verify_token(self):
#         return self.token

    def genlbc(self):

        hint = "none"
        question = "What day(s) do you visit this place/area most often? "
        if self.geoprocdata.poi_main_df['weekday_only'].ix[self.geoprocdata.selected_poi_for_freq_query]==True:
            self.geoprocdata.valid_answer.append('weekdays')
        elif self.geoprocdata.poi_main_df['weekend_only'].ix[self.geoprocdata.selected_poi_for_freq_query]==True:
            self.geoprocdata.valid_answer.append('weekends')
        self.geoprocdata.valid_answer.append(calendar.day_name[self.geoprocdata.most_freq_dotw_for_pot].lower())

        self.challenge = [question, hint]
        self.gen_token = self.tokemon()

        return self.challenge, self.gen_token

    def passlbc(self,answer,passed_token):

        # Validate token first

        if passed_token != self.token:
            return 0
        else:
            if answer in self.geoprocdata.valid_answer:
                return 1
            if answer not in self.geoprocdata.valid_answer:
                return -1

    def lbc_auth(self,challenge_response,token):

        if self.tries>0:
            self.tries -= 1
            response = chal_answer.lower().translate(None, string.punctuation)
            attempt = self.passlbc(response,token)
            if attempt==1:
                print "Login OK - You have been identified as an authorized user."
                self.tries=0
            elif attempt==-1:
                print "Authenticated failed. Please try again."
            elif attempt==0:
                print "Insecure authentication attempt detected. System may be compromised."
                self.tries=0
        else:
            print "Authentication Failed"

    def get_map_url(self):
        return self.geoprocdata.getmapurl(self.geoprocdata.mean_lat(), self.geoprocdata.mean_lng())

    def use_try(self):
        self.tries -= 1

    def check_tries(self):
        return self.tries

    def val_answer(self):
        return self.geoprocdata.valid_answer

#-------- ROUTES GO HERE -----------#

# This method takes input via an HTML page
@app.route('/auth')
def page():
    htmlstring = "<body>\
    <iframe width=\"400px\" height=\"300px\" src=\"%s\" frameborder=\"0\" allowfullscreen=\"\"></iframe>\
          <form action = \"http://localhost:11000/chauth\" method = \"POST\">\
             <p>%s<input type =\"text\" name = \"c_answer\" /></p>\
             <p><input type =\"hidden\" name = \"token\" value=\"%s\"/></p>\
             <p><input type = \"submit\" value = \"submit\" /></p>\
          </form>\
       </body>"%(newauth.get_map_url(), chal_prompt[0], chal_token)
    return htmlstring

#    with open("auth.html", 'r') as viz_file:#
#        return viz_file.read()

@app.route('/chauth', methods=['POST', 'GET'])
def result():

    if flask.request.method == 'POST':

        inputs = flask.request.form
        chal_answer = inputs['c_answer']
        token = inputs['token']

#       return "%s,%s,%s"%(chal_answer,token,newauth.validate_answer())
        if newauth.check_tries() > 0:

            if str(newauth.passlbc(chal_answer.lower(),token)) == "1":
                return "<body><p>Auth OK!</p></body"
            elif str(newauth.passlbc(chal_answer.lower(),token)) == "-1":
                newauth.use_try()
                return redirect("http://127.0.0.1:11000/auth")
            elif str(newauth.passlbc(chal_answer,token)) == "0":
                return "<body><p>System Compromised</p></body"
        else:
            return "<body><p>Maximum Tries Exceeded</p></body>"

#-------- MAIN RUN -----------#
if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '11000'

    newauth = lbc('LocationHistory-20161121.json')
    chal_prompt, chal_token = newauth.genlbc()

    app.run(HOST, PORT)
