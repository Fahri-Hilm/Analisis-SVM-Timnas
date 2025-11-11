def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_csv(dataframe, file_path):
    """Save a pandas DataFrame to a CSV file."""
    import pandas as pd
    dataframe.to_csv(file_path, index=False)

def clean_text(text):
    """Clean and preprocess the input text."""
    import re
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def get_sentiment_distribution(sentiment_series):
    """Get the distribution of sentiments as a dictionary."""
    return sentiment_series.value_counts(normalize=True).to_dict()