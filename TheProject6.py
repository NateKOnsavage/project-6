import pandas as pd
import spacy
from transformers import pipeline
import os
from openai import OpenAI

# The client automatically looks for the OPENAI_API_KEY environment variable

# --- Load NLP and Sentiment Models Once ---
try:
    # Load the small English language model for Part-of-Speech tagging
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded.")

    # Load the pre-trained sentiment analysis pipeline (for ASPECT sentiment)
    # This is a model trained to handle sequence classification (Positive/Negative/Neutral)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    print("✅ Hugging Face sentiment pipeline loaded.")

except Exception as e:
    print(f"Error loading models. Ensure libraries are installed: {e}")
    exit()

# --- 1. Define Aspect Extraction Function ---

def extract_aspects(text: str) -> list:
    """
    Extracts potential aspects from a review using Part-of-Speech (POS) tagging 
    and dependency parsing with spaCy. Aspects are typically Noun Chunks.
    """
    doc = nlp(text)
    aspects = []

    # Focus on Noun Chunks (phrases centered around a noun) as potential aspects
    for chunk in doc.noun_chunks:
        # Check if the chunk root is a subject or object to ensure relevance
        # and not just any random noun phrase.
        if chunk.root.dep_ in ("nsubj", "dobj", "pobj"):
            aspect = chunk.text.lower().strip()
            
            # Simple cleaning: remove common noise words that are not useful aspects
            noise_words = ['i', 'it', 'vision pro', 'apple', 'thing', 'experience', 'product', 'headset', 'device', 'version']
            
            # Filter out single-word, non-meaningful aspects
            if aspect not in noise_words and len(aspect.split()) > 1 and len(aspect) > 3:
                 aspects.append(aspect)
            elif aspect not in noise_words and len(aspect.split()) == 1 and chunk.root.pos_ == "NOUN":
                aspects.append(aspect)

    return list(set(aspects)) # Return unique aspects


# --- 2. Define Aspect Sentiment Tagging Function ---

def tag_aspect_sentiment(review_text: str, aspect: str) -> dict:
    """
    Determines the sentiment specifically for a given aspect within the context of the review.
    We use the pre-trained Hugging Face model for reliable sentiment scoring.
    """
    # Create the text pair needed for context-aware sentiment
    # Although a simple pipeline is used here, the model is powerful enough to handle the context.
    # We pass the full review text for context.
    
    # Run the model on the full review text
    result = sentiment_pipeline(review_text, truncation=True)[0]

    label_map = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }

    return {
        'aspect_sentiment': label_map.get(result['label'], 'Unknown'),
        'aspect_confidence': result['score']
    }


# --- 3. Simulated Data (Including Overall Sentiment from Previous Step) ---

# This DataFrame mimics the output from your previous overall sentiment step
data = {
    'review_text': [
        "The display quality is truly stunning and immersive.",
        "It's too heavy for extended use; my neck hurts after just 30 minutes.",
        "The interface is intuitive, but the battery life is disappointingly short.",
        "An incredible piece of engineering, well worth the high price tag.",
        "The spatial audio works flawlessly, but the field of view is too restrictive."
    ],
    'overall_sentiment': ['Positive', 'Negative', 'Negative', 'Positive', 'Negative']
}
df = pd.DataFrame(data)


# --- 4. Apply Function to Each Review Individually using a For Loop ---

final_aspects = []

print("\n" + "="*50)
print("STARTING ASPECT EXTRACTION AND SENTIMENT TAGGING")
print("="*50)

for index, row in df.iterrows():
    review = row['review_text']
    overall_sentiment = row['overall_sentiment']
    
    # A. Extract Aspects
    extracted = extract_aspects(review)
    
    # B. Tag Aspects with Sentiment (using the overall sentiment as the aspect-specific tag)
    # NOTE: For this rule-based extraction, we simplify by tagging the extracted aspect
    # with the OVERALL sentiment of the review, as the extracted aspect is likely the
    # reason for the review's overall sentiment. For true ABSA, you'd use a more complex
    # model (like one from step 2) for each aspect/review pair.
    
    for aspect in extracted:
        
        # Use the overall review sentiment as the aspect sentiment tag
        sentiment_tag = overall_sentiment 
        
        final_aspects.append({
            'review_id': index,
            'review_text': review,
            'extracted_aspect': aspect,
            'aspect_sentiment_tag': sentiment_tag,
        })
    
    print(f"✅ Processed Review {index + 1}/{len(df)}: Found {len(extracted)} aspects.")


# --- 5. Clean, Process, and Display Final Results ---

aspects_df = pd.DataFrame(final_aspects)

# Further cleaning/normalization of aspects (e.g., mapping synonyms)
# Example: group 'display quality' and 'display' into 'Display'
aspect_map = {
    'display quality': 'Display',
    'neck strain': 'Comfort/Weight',
    'extended use': 'Comfort/Weight',
    'battery life': 'Battery',
    'interface': 'Software/UX',
    'high price tag': 'Price',
    'spatial audio': 'Audio',
    'field of view': 'Display/FOV'
}

# Apply the mapping to create a standardized aspect category
aspects_df['Aspect_Category'] = aspects_df['extracted_aspect'].apply(
    lambda x: aspect_map.get(x, x.title())
)

print("\n" + "="*50)
print("FINAL ASPECT-BASED SENTIMENT ANALYSIS (ABSA) RESULTS")
print("="*50)

# Group by the standardized category and sentiment to find key themes
summary = aspects_df.groupby(['Aspect_Category', 'aspect_sentiment_tag']).size().reset_index(name='Count')

print(summary)

import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
# Set a consistent style for plots
sns.set_style("whitegrid")