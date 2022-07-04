import pandas as pd
from sklearn.model_selection import train_test_split


def trim(text, n_words=200):
    def f(x, n):
        x = x.split(maxsplit=n)
        x = ' '.join(x[:n])
        return x
    return text.apply(lambda x: f(x, n_words))


def normalize(text):
    text = text.str.lower()
    text = text.str.replace(r'[^A-Za-z0-9]+', ' ', regex=True)
    text = text.str.replace('\s{2,}', ' ', regex=False)
    return text


def split_data(df):
    # Split according to label
    df_real = df[df['label'] == 0]
    df_fake = df[df['label'] == 1]

    # Train-test split
    df_real_full_train, df_real_test = train_test_split(df_real, train_size = 0.8, random_state = 1)
    df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size = 0.8, random_state = 1)

    # Train-valid split
    df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size = 0.8, random_state = 1)
    df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size = 0.8, random_state = 1)

    # Concatenate splits of different labels
    df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
    df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)
    return df_train, df_valid, df_test


def main(input_path, output_path):
    # Read file
    df_raw = pd.read_csv(input_path + '/news.csv')
    
    # Create columns
    df_raw['label'] = (df_raw['label'] == 'FAKE').astype('int')
    df_raw['titletext'] = df_raw['title'] + ". " + df_raw['text']
    df_raw = df_raw.reindex(columns=['label', 'title', 'text', 'titletext'])

    # Drop rows with empty text
    df_raw.drop( df_raw[df_raw.text.str.len() < 5].index, inplace=True)

    # Normalize text
    df_raw['text'] = normalize(df_raw['text'])
    df_raw['titletext'] = normalize(df_raw['titletext'])

    # Trim text
    df_raw['text'] = trim(df_raw['text'])
    df_raw['titletext'] = trim(df_raw['titletext'])

    # Split data and save files
    df_train, df_valid, df_test = split_data(df_raw)
    df_train.to_csv(output_path + '/train.csv', index=False)
    df_valid.to_csv(output_path + '/valid.csv', index=False)
    df_test.to_csv(output_path + '/test.csv', index=False)
