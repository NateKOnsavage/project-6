import pandas as pd
from transformers import pipeline

# --- 1. Define the Sentiment Analysis Function (using Hugging Face RoBERTa) ---

def analyze_sentiment_hf(text: str) -> dict:
    """
    Performs sentiment analysis on a SINGLE text string using a pre-trained
    Hugging Face RoBERTa model fine-tuned for general sentiment.

    Args:
        text: A single customer review (string).

    Returns:
        A dictionary containing the sentiment results for the single review.
    """
    # Define the pipeline. It's defined inside to make the function self-contained,
    # but for true efficiency in a production environment, initialize it once globally.
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    except Exception as e:
        print(f"Error loading the model. Please check installation: {e}")
        return {'sentiment': 'ERROR', 'confidence_score': 0.0}

    # Process the single text
    result = sentiment_pipeline(text, truncation=True)[0]

    # Map the model's labels to human-readable labels
    label_map = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }

    return {
        'sentiment': label_map.get(result['label'], 'Unknown'),
        'confidence_score': result['score']
    }


# --- 2. Data Simulation (Replace with your actual DB loading) ---
# NOTE: In your actual project, replace this section with code to load your feedback.db data.

# Simulated reviews (representing a subset of your 80-100 reviews)
simulated_reviews = [
    "The display is absolutely stunning, making the high price almost worth it.",
    "It's too heavy for extended use; my neck hurts after just 30 minutes.",
    "The immersive experience is revolutionary, but the app selection is sparse.",
    "I found the setup process confusing and buggy, but once running, it's magical.",
    "Just an 'okay' product for now; I'll wait for the second generation.",
    "The eye-tracking is flawless and incredibly intuitive.",
    "Battery life is a serious bottleneck; I constantly need to charge it.",
    "The resolution makes the virtual desktop feature completely unusable."
]

# Create a DataFrame to hold the simulated data (mimicking what you'd load from DB)
df = pd.DataFrame({'review_text': simulated_reviews})


# --- 3. Apply Function to Each Review Individually using a For Loop ---

sentiment_results = []
print(f"Starting sentiment analysis on {len(df)} reviews...")

# Iterate over each review text using the DataFrame's iterrows()
for index, row in df.iterrows():
    review = row['review_text']

    # Apply the function to the individual review text
    analysis = analyze_sentiment_hf(review)

    # Store the result
    result_dict = {
        'review_text': review,
        'sentiment': analysis['sentiment'],
        'confidence_score': analysis['confidence_score']
    }
    sentiment_results.append(result_dict)

    print(f"✅ Processed review {index + 1}/{len(df)}: Sentiment = {analysis['sentiment']}")


# --- 4. Aggregate and Display Final Results ---

# Convert the list of results into a DataFrame
results_df = pd.DataFrame(sentiment_results)

# Merge the results back into the original DataFrame (if needed, or just use results_df)
final_df = df.merge(results_df, on='review_text', how='left')

print("\n" + "="*50)
print("FINAL SENTIMENT ANALYSIS RESULTS")
print("="*50)
print(final_df[['review_text', 'sentiment', 'confidence_score']])

print("\nSENTIMENT DISTRIBUTION:")
print(final_df['sentiment'].value_counts())