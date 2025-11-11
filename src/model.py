class SentimentSVMModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        self.vectorizer = None
        self.label_encoder = None

    def train(self, X, y):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.model = make_pipeline(TfidfVectorizer(max_features=5000), SVC(kernel=self.kernel, C=self.C, gamma=self.gamma))
        self.model.fit(X, y_encoded)

    def predict(self, texts):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        return self.label_encoder.inverse_transform(self.model.predict(texts))

    def get_feature_importance(self, top_n=20):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        
        feature_names = self.model.named_steps['tfidfvectorizer'].get_feature_names_out()
        coefs = self.model.named_steps['svc'].coef_[0]
        top_features = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {'top_features': top_features}