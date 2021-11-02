import functions as mf
import model as md
import argparse
import os


PATH = '/home/dietmar/spiced/random-forest-fennel-student-code/week_04'

text = input('Type in your lyrics: ')

parser = argparse.ArgumentParser(description = 'Type in two artists and the number of songs to evaluate and type in your self written lyrics and the program will show you which artist could have wrote your lyrics')

parser.add_argument("-a1", "--artist_1",
                     help="Takes the first artist",
                     type=str,
                     default='Coldplay')

parser.add_argument("-a2", "--artist_2",
                     help="Takes the second artist",
                     type=str,
                     default='Nirvana')

parser.add_argument("-n", "--num",
                     help="Takes the number of songs for each artist to consider",
                     type=int,
                     default=100)

args = parser.parse_args()


# loads lyrics data from genius if file does not exist yet
# note: loading new lyrics takes time
if mf.find(args.artist_1, PATH):
    mf.get_artist_save_lyrics(args.artist_1, args.num)
elif mf.find(args.artist_2, PATH):
    mf.get_artist_save_lyrics(args.artist_2, args.num)


list_of_lyrics = mf.get_lyrics_list(args.artist_1, args.num)
df_coldplay = mf.create_dataframe(args.artist_1, list_of_lyrics)

list_of_lyrics = mf.get_lyrics_list(args.artist_2, args.num)
df_nirvana = mf.create_dataframe(args.artist_2, list_of_lyrics)

df = mf.concat_dataframes(df_coldplay, df_nirvana)

df = md.check_non_ascii(df)

corpus = md.corpus_maker(df)

clean_corpus = md.corpus_cleaner(corpus)

labels = md.creating_labels(args.artist_1, args.artist_2, args.num)

df_vec = md.vectorize_and_normalize_corpus(clean_corpus, labels)

md.ml_model(clean_corpus, labels, text)


