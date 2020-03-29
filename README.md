# Youtube Comments Classifier

Trained on the dataset provided by the Kaggle Toxic Comments dataset, using TD-IDF vectorizers in `sklearn`.

There is a notebook which trains a classification model, and Python script which expects certain arguments passed in.

The comment classes are: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate.

### Example usage

```
python youtube_comments/youtube.py --key_file personal_key.json --url https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

The result of this will be
* Comment class fractions & counts printed to the screen 
* This summary will be saved by default to the base directory of this repo

To specify somewhere else for this data to be saved, use the argument:
```
--data_path=~/example/directory/output_folder
```

To save the entire list of comments on a YouTube video, use the argument:
```
--save_raw=True
```
