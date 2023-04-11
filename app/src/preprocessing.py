import pandas as pd

# Preprocessing track genres
tracks = pd.read_csv('data/tracks.csv', delimiter=',', header=[0, 1])

# raw_genres file to map each genre with genre_id
genres = pd.read_csv('data/raw_genres.csv', delimiter=',')

track_genres = tracks[('track', 'genres')]
track_top_genre = tracks[('track', 'genre_top')]
track_all_genres = tracks[('track', 'genres_all')]

processed_tracks_metadata = pd.DataFrame({'genre_top': track_top_genre, 'genres': track_genres, 'genres_all': track_all_genres})
processed_tracks_metadata.index.name = 'track_id'

# Map genre_title -> genre_id
processed_tracks_metadata['genre_top'] = processed_tracks_metadata['genre_top'].apply(lambda x: genres.loc[genres['genre_title'] == x, 'genre_id'].values)
print(processed_tracks_metadata.head())

# Save a CSV file with track_id, top_genre, genres and total genres of each track.
processed_tracks_metadata.to_csv('data/processed/track_genres.csv')