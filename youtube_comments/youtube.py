"""
Script to fetch Youtube comments from a URL and summarise them in terms of toxicity. This can 
be used to categorise videos where toxic commenters congregate, or to flag up users who are
posting identity hate comments.
"""

import os
import pickle
import getpass
import csv
import re
import argparse

import pandas as pd
import numpy as np 

from scipy.sparse import hstack

# Google API imports
import google.oauth2.credentials

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


##################################################################################################################
# -------------------------------------------------------------------------------------------------------------- #
#                                               CONSTANTS                                                        #
# -------------------------------------------------------------------------------------------------------------- #
##################################################################################################################


COMMENT_CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

MODELS_DICT_LOCATION = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../models/models_dict.pkl'))

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

API_SERVICE_NAME = 'youtube'

API_VERSION = 'v3'

# TODO: handle this secrets file better. Command line argument? Or prompt the user if the key file doesn't exist already?
SECRETS_FILENAME = 'client_secret_1049876915637-7ia95c7rg5teak6crcuodies22keluuh.apps.googleusercontent.com.json'
CLIENT_SECRETS_FILE = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', SECRETS_FILENAME))



##################################################################################################################
# -------------------------------------------------------------------------------------------------------------- #
#                                               FUNCTIONS                                                        #
# -------------------------------------------------------------------------------------------------------------- #
##################################################################################################################

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=dQw4w9WgXcQ')

    parser.add_argument(
        '--data_path', 
        type=str, 
        default=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    )
    
    parser.add_argument('--save_raw', type=str2bool, default=False)

    retval, _ = parser.parse_known_args()

    return vars(retval)


def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            credentials = flow.run_console()
 
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)


def get_video_id_from_url(url):
    return re.split('v=', url)[-1]


def get_video_comments_from_url(url, service, **kwargs):
    
    video_id = get_video_id_from_url(url)
    
    comments = []
    results = service.commentThreads().list(
        part='snippet', videoId=video_id, textFormat='plainText', **kwargs
    ).execute()

    while results:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(
                part='snippet', videoId=video_id, textFormat='plainText', **kwargs
            ).execute()
        else:
            break
    
    return comments


def classify_comments(comments, word_vectorizer, char_vectorizer, models, probability=False):
    """
    :param comments: an array of strings, the raw data to score
    """
    word_features = word_vectorizer.transform(comments)
    char_features = char_vectorizer.transform(comments)
    combined_features = hstack([char_features, word_features])
    
    predictions = {}
    for class_name, model in models.items():
        if probability:
            # Take the positive class probability prediction
            class_prediction = model.predict_proba(combined_features)[1]
        else:
            class_prediction = model.predict(combined_features)
            
        predictions[class_name] = class_prediction
    
    return pd.DataFrame(predictions)


def summarise_comments_data(comments_data):
    summary = comments_data[COMMENT_CATEGORIES].agg(['mean', 'sum'], axis=0).astype(np.float32)
    # TODO: add other types of summary
    
    # Can extract a list of all users who are seen to be spreading identity hate?
    
    return summary



if __name__ == '__main__':
    # To run locally (not in a production environment), we
    # disable OAuthlib's HTTPs verification.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()
    
    args = parse_args()
    
    youtube_comments = get_video_comments_from_url(args['url'], service)
    comments_data = pd.DataFrame({
        'comment': youtube_comments
    })
    
    video_id = get_video_id_from_url(args['url'])
    if len(youtube_comments) == 0:
        raise ValueError(f'The video {video_id} currently has no comments!')
    
    with open(MODELS_DICT_LOCATION, 'rb') as f:
        models_dict = pickle.load(f)
        
    word_vectorizer = models_dict['word_vectorizer']
    char_vectorizer = models_dict['char_vectorizer']
    models = models_dict['models']
    
    comments_data = classify_comments(
        comments_data['comment'], 
        word_vectorizer, 
        char_vectorizer, 
        models
    )
    
    summary = summarise_comments_data(comments_data)
    
    print(summary)
    
    summary.to_csv(f'{args["data_path"]}/{video_id}.csv', header=True)
    
    if args['save_raw']:
        comments_data.to_csv(f'{args["data_path"]}/{video_id}_all_comments.csv')
    
    