import pandas as pd
from transformers import pipeline

class NLPEngine:
    def __init__(self):

        # Pre-trained model for sentiment analysis (Transformer)
        print("Model loading...")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def get_sentiment_batch(self, texts):
        """It retrieves the text list and returns the sentiment results."""
        # Batch processing for performance
        cleaned_texts = [str(t)[:512] for t in texts] # 512 character limit
        results = self.sentiment_analyzer(cleaned_texts)
        return results
    
if __name__ == "__main__":
    engine = NLPEngine()
    df = engine.load_data('data/clean_reviews.csv')
    
    # 1st Group: General 200 row (Mostly positive)
    sample_df = df.head(200).copy() # Example Application (Let's test the first 200 lines for speed)
    sample_results = engine.get_sentiment_batch(sample_df['full_text'].tolist())
    sample_df['sentiment_label'] = [r['label'] for r in sample_results]
    sample_df['sentiment_score'] = [r['score'] for r in sample_results]
    
    for i in range(5):
        print(f"Sample Text: {sample_df['full_text'].iloc[i]} \nSentiment Result: {sample_results[i]}")
        print("-"*50)

    # 2nd Group: Low-rated reviews (Rating <= 2)
    potential_negatives = df[df['Rating'] <= 2].copy()
    print(f"Total number of potential negatives: {len(potential_negatives)}")
    negative_df = potential_negatives.head(50).copy() # Test on the first 50 potential negatives
    neg_results = engine.get_sentiment_batch(negative_df['full_text'].tolist())
    negative_df['sentiment_label'] = [r['label'] for r in neg_results]
    negative_df['sentiment_score'] = [r['score'] for r in neg_results]
    
    print("\n--- LOW-RATED REVIEWS MODEL ANALYSIS ---")
    for i in range(5):
        row = negative_df.iloc[i]
        print(f"Rating: {row['Rating']} | Model Prediction: {row['sentiment_label']}")
        print(f"Review: {row['full_text']}...")
        print("-" * 30)

    # 3rd Group: Combine both groups for overall insight
    total_df = pd.concat([sample_df, negative_df], ignore_index=True)
    print(f"Total Reviewed: {len(total_df)}")

    # Statistics:
    avg_score = total_df['sentiment_score'].mean()
    positive_ratio = (total_df['sentiment_label'] == "POSITIVE").mean()
    negative_ratio = (total_df['sentiment_label'] == "NEGATIVE").mean()
    
    print("\n--- SUMMARY RESULTS ---")
    print(f"Average Sentiment Score: {avg_score:.4f}")
    print(f"Positive Ratio: {positive_ratio:.4f}")
    print(f"Negative Ratio: {negative_ratio:.4f}")  
    print(f"Insight Score: {positive_ratio - negative_ratio:.4f}")

    # Anomalies 
    # Are there any reviews with Rating 1 that the model predicts as POSITIVE?
    fake_positives = negative_df[negative_df['sentiment_label'] == 'POSITIVE']
    print(f"\nHarsh Criticism Misclassified as Positive (Anomaly): {len(fake_positives)}")
    if len(fake_positives) > 0:
        for i in range(5):
            print(f"Example of Anomaly: Rating {fake_positives.iloc[i]['Rating']} | Review: {fake_positives.iloc[i]['full_text']}")