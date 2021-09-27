from joblib import load

class PredictSentiment():

    def __init__(self):
        """
        Initialize this object by loading our classifier model for sentiment prediction.
        """
        self.model = load("../models/sentiment_classifier.joblib")
    
    def get_sentiment(self, sentence: str) -> dict:
        """
        Returns a dictionary containg a key value pair sentinment of 1 or 0 depending on if the sentence submitted has positive (1) or negative (0) sentiment.
        The object also has a key value pair containing the confidence of the prediction as well.
        Dict:
            - positive: Bool
            - confidence: Float
        """
        sent_list = [sentence]
        return {
            "positive" : self.model.predict(sent_list) == 1,
            "confidence": self.model.predict_proba(sent_list)[0][1]
        } 

