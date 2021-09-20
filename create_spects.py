import numpy as np
import librosa as lbs
import os

genres = os.listdir("genres_original")
genres.remove(".DS_Store")
genre_dict = {genres[i]: i for i in range(len(genres))}

x, y = np.zeros((999, 128, 647)), np.zeros((999, 1))
count = 0
for genre in genres:
	for image in os.listdir(f"genres_original/{genre}"):
		print(image)
		if image == "jazz.00054.wav":
			print("skipping", image)
			continue
		y[count] = np.array(genre_dict[genre])
		time_series, _ = lbs.load(f"genres_original/{genre}/{image}")
		spect = lbs.feature.melspectrogram(y=time_series, hop_length=1024, n_fft=2048)[:, :647]
		if spect.shape[-1] < 647:
			amount = 647 - spect.shape[-1]
			spect = np.concatenate((spect, np.zeros((128, amount))), axis=1)
		x[count] = spect
		count += 1

x,y = np.array(x), np.array(y)
np.save("genres_numpy.npy", x)
np.save("genre_labels.npy", y)
