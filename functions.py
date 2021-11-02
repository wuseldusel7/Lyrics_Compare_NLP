import requests
import pandas as pd
import numpy as np
import re
import lyricsgenius
import json
import os


GENIUS_ACCESS_TOKEN = 'UWHySPRfpRkA4A6ao-Mn33Di_NmC6BSlHi-kZHXxBj5mmmnNYnNBebOK-PdHNqhi'

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)


genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = False # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Remix)", "(Live)", "()"] # Exclude songs with these words in their title

def get_artist_save_lyrics(singer, num_of_songs):
    '''
    This function scrapes for various information of specific artists/singers and 
    saves the information into a json-file. 
    num_of_songs is the number of songs to scrape for where the first is the most popular one
    '''
    artist = genius.search_artist(singer, max_songs = num_of_songs)
    artist.save_lyrics()
    return


def get_lyrics_list(singer, num_of_songs):
    '''
    This extracts the lyrics of the most popular songs.
    num_of_songs is the number of songs to get the lyrics from where the first is the most popular song
    '''
    singer = singer.replace(' ','')
    input_file=open(f'Lyrics_{singer}.json', 'r')
    json_decode=json.load(input_file)

    lst = []
    for i in range(num_of_songs):
        lst.append(json_decode['songs'][i]['lyrics'].replace('\n', ' '))

    return lst


def create_dataframe(singer, lst):
    '''
    This creates an dataframe with numerical indexing 
    and two columns with lyrics and artist
    '''
    df = pd.DataFrame()
    df['lyrics'] = lst
    df['artist'] = singer
    
    return df


def concat_dataframes(df1, df2):
    '''
    This concats two dataframes of the same form together
    '''
    df = pd.concat([df1, df2])
    return df.reset_index().drop('index', axis=1)


def find(name, path):
    '''
    This takes a file name and it's path and returns True if the filename exists and False if not
    '''
    for root, dirs, files in os.walk(path):
        if name in files:
            return True
        else:   return False



if __name__ == "__main__":
    pass