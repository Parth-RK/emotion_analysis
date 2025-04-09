import re

def clean_text(text):
    """
    Basic text cleaning: lowercase, remove user mentions, URLs, and excessive whitespace.
    """
    text = str(text).lower()
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs (http:// or https://)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove non-alphanumeric characters (keeping basic punctuation that might be relevant)
    # text = re.sub(r'[^a-z0-9\s.,!?\'"]', '', text) # Optional: more aggressive cleaning
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Example usage (optional)
if __name__ == '__main__':
    sample_text = "@tiffanylue i know  i was listenin to bad habit earlier and i started freakin at his part =[ http://example.com"
    cleaned = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}") # Output: i know i was listenin to bad habit earlier and i started freakin at his part =[