import pandas as pd

def create_balanced_dataset(input_path, output_path, n_samples=10):
    """
    Creates a balanced dataset with n samples from each emotion category.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the balanced dataset
        n_samples (int): Number of samples to take from each emotion category
    
    Returns:
        pd.DataFrame: The balanced dataset
    """
    print(f"Reading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"Original dataset shape: {df.shape}")
    
    # Check if required columns exist
    required_cols = ['emotion', 'content']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Check for sentiment column as alternative to emotion
        if 'sentiment' in df.columns and 'emotion' in missing_cols:
            print("Found 'sentiment' column instead of 'emotion', using that column")
            df = df.rename(columns={'sentiment': 'emotion'})
            missing_cols.remove('emotion')
        
        # If still missing columns, exit
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Expected format: tweet_id,emotion,content")
            return None
    
    # Drop NaN values
    df = df.dropna(subset=['emotion', 'content']).reset_index(drop=True)
    
    # Get emotion counts
    emotion_counts = df['emotion'].value_counts()
    print("\nEmotion counts in original dataset:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")
    
    # Create balanced dataset
    balanced_dfs = []
    
    for emotion in emotion_counts.index:
        emotion_df = df[df['emotion'] == emotion]
        
        # Take n samples or all if fewer than n
        sample_size = min(n_samples, len(emotion_df))
        if sample_size < n_samples:
            print(f"Warning: Emotion '{emotion}' has only {sample_size} samples (requested {n_samples})")
        
        # Sample without replacement
        sampled = emotion_df.sample(n=sample_size, random_state=42)
        balanced_dfs.append(sampled)
    
    # Combine all emotions
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced dataset shape: {balanced_df.shape}")
    print("\nEmotion counts in balanced dataset:")
    for emotion, count in balanced_df['emotion'].value_counts().items():
        print(f"  {emotion}: {count}")
    
    print(f"\nSaving balanced dataset to {output_path}...")
    balanced_df.to_csv(output_path, index=False)
    print("Done!")
    
    return balanced_df

if __name__ == "__main__":
    input_path = 'data/emotion_data.csv'
    output_path = 'data/emotion_data_lite.csv'
    
    create_balanced_dataset(input_path, output_path, n_samples=50)