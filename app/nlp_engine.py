import pandas as pd
# import mlflow
from transformers import pipeline

class NLPEngine:
    def __init__(self):
        # Pre-trained model for sentiment analysis (Transformer)
        print("Model loading...")
        # We use a more advanced RoBERTa model for better context understanding instead of a simple DistilBERT. 
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def load_data(self, file_path):
        """Loads the cleaned dataset."""
        df = pd.read_csv(file_path)
        return df

    def get_sentiment_batch(self, texts):
        """It retrieves the text list and returns the sentiment results."""
        # Batch processing for performance
        cleaned_texts = [str(t)[:512] for t in texts] # 512 character limit for Transformer
        results = self.sentiment_analyzer(cleaned_texts)
        # Standardize labels to lowercase to avoid case-sensitivity issues
        for res in results:
            res['label'] = res['label'].lower()
        return results
        # label : positive, negative, neutral
        # score : How model is confident about the prediction (0 to 1)

    def find_anomalies(self, df):
        """It finds the inconsistencies between Rating and Sentiment."""
        # Detect reviews with high rating but negative sentiment
        high_rating_neg_sent = df[(df['Rating'] >= 4) & (df['sentiment_label'] == 'negative')]
        # Detect reviews with low rating but positive sentiment
        low_rating_pos_sent = df[(df['Rating'] <= 2) & (df['sentiment_label'] == 'positive')]
        return pd.concat([high_rating_neg_sent, low_rating_pos_sent])
    

# Helper function for pandas apply (Consistency Score Calculation)
def calculate_consistency(row):
    """Calculates how much the Rating and Sentiment match."""
    rating = row['Rating']
    label = str(row['sentiment_label']).lower()
    score = row['sentiment_score']
    is_shouting = row['is_shouting']
    is_intense = row['is_intense']

    consistency = 0.5 # Slight Conflict / Neutral cases
    # Scenario 1: Consistent Positive (Rating 4-5 and Positive) OR Consistent Negative (Rating 1-2 and Negative)
    if (rating >= 4 and label == 'positive') or (rating <= 2 and label == 'negative'):
        consistency = score
    # Scenario 2: Total Conflict (Rating 5 but Negative / Rating 1 but Positive)
    elif (rating == 5 and label == 'negative') or (rating == 1 and label == 'positive'):
        consistency = 1 - score # Consistency is very low
        # Special Case: Intense Factor
        if is_intense: 
            consistency = 1 - score # Consistency is very low

    return consistency

if __name__ == "__main__":

    engine = NLPEngine()
    df = engine.load_data('data/clean_reviews.csv')
    
    # 1st Group: General 200 row (Mostly positive)
    sample_df = df.head(200).copy()  # test the first 200 lines for speed
    sample_results = engine.get_sentiment_batch(sample_df['full_text'].tolist())
    sample_df['sentiment_label'] = [r['label'] for r in sample_results]
    sample_df['sentiment_score'] = [r['score'] for r in sample_results]
    
    print("\n--- SAMPLE GENERAL REVIEWS ---")
    for i in range(3):
        print(f"Text: {sample_df['full_text'].iloc[i][:300]}... \nResult: {sample_results[i]}")
        print("-" * 30)

    # 2nd Group: Low-rated reviews (Rating <= 2)
    potential_negatives = df[df['Rating'] <= 2].copy()
    print(f"Total number of potential negatives in dataset: {len(potential_negatives)}")
    
    negative_df = potential_negatives.head(200).copy() # Test on the first 250 potential negatives
    neg_results = engine.get_sentiment_batch(negative_df['full_text'].tolist())
    negative_df['sentiment_label'] = [r['label'] for r in neg_results]
    negative_df['sentiment_score'] = [r['score'] for r in neg_results]
    
    print("\n--- LOW-RATED REVIEWS MODEL ANALYSIS ---")
    for i in range(3):
        row = negative_df.iloc[i]
        print(f"Rating: {row['Rating']} | Prediction: {row['sentiment_label']}")
        print(f"Review: {row['full_text'][:300]}...")
        print("-" * 30)

    # 3rd Group: Combine both groups for overall insight
    total_df = pd.concat([sample_df, negative_df], ignore_index=True)

  # Let's check if 'shouting' correlates with 'negative' sentiment
    shouting_negatives = total_df[(total_df['is_shouting'] == True) & (total_df['sentiment_label'] == 'negative')]
    
    print("\n--- SHOUTING ANALYSIS (Intensity Check) ---")
    print(f"Number of shouting reviews: {total_df['is_shouting'].sum()}")
    print(f"Shouting & Negative overlap: {len(shouting_negatives)}")
    
    if len(shouting_negatives) > 0:
        print("Example of a 'Shouting' Negative Review:")
        print(shouting_negatives['full_text'].iloc[0][:200])

    # Calculate consistency score for each row
    total_df['consistency_score'] = total_df.apply(calculate_consistency, axis=1)

    # Statistics Calculation
    avg_score = total_df['sentiment_score'].mean()
    pos_ratio = (total_df['sentiment_label'] == "positive").mean()
    neg_ratio = (total_df['sentiment_label'] == "negative").mean()
    neu_ratio = (total_df['sentiment_label'] == "neutral").mean()
    
    print("\n" + "="*40)
    print("SUMMARY RESULTS")
    print("="*40)
    print(f"Total Reviewed: {len(total_df)}")
    print(f"Positive: %{pos_ratio*100:.2f} | Negative: %{neg_ratio*100:.2f} | Neutral: %{neu_ratio*100:.2f}")
    print(f"Avg Sentiment Confidence: {avg_score:.4f}") # Average confidence of the model's predictions
    print(f"Avg Consistency Score: {total_df['consistency_score'].mean():.4f}") # How well the ratings and sentiments align on average
    print(f"Insight Score (P-N): {pos_ratio - neg_ratio:.4f}")

    # Anomalies detection 
    # Are there any reviews with Rating 1-2 that the model predicts as positive?
    anomalies = engine.find_anomalies(total_df)
    print(f"\nHarsh Criticism Misclassified (Anomaly): {len(anomalies)}")
    
    if len(anomalies) > 0:
        print("\n--- TOP ANOMALIES (SUSPICIOUS REVIEWS) ---")
        # We sort by consistency_score to see the most conflicting ones first
        anomalies_sorted = anomalies.sort_values(by='sentiment_score', ascending=False)
        for i in range(min(5, len(anomalies_sorted))):
            row = anomalies_sorted.iloc[i]
            print(f"Rating: {row['Rating']} | Model: {row['sentiment_label']} | Review: {row['full_text'][:400]}...")
            print("-" * 20)


    # Export results for next steps (Vector DB)
    total_df.to_csv('data/analyzed_reviews.csv', index=False)