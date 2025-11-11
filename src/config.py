# Configuration settings for the sentiment analysis project

# API keys and settings
API_KEY = "YOUR_YOUTUBE_API_KEY"

# Search keywords for YouTube data collection
SEARCH_KEYWORDS = [
    "Indonesia Piala Dunia",
    "Timnas Indonesia",
    "Sepakbola Indonesia",
    "Kegagalan Piala Dunia"
]

# Data collection settings
MAX_VIDEOS_PER_QUERY = 5
MAX_COMMENTS_PER_VIDEO = 100

# SVM model parameters
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)