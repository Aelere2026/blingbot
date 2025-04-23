# from sklearn.model_selection import StratifiedShuffleSplit

# # Load the dataset
# df = pd.read_csv('data.csv')

# # Bin the price column into 10 quantile-based categories
# df['price_bin'] = pd.qcut(df['price'], q=10, labels=False)

# # Stratified split: 80% train, 20% validation
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_idx, val_idx in split.split(df, df['price_bin']):
#     train_df = df.loc[train_idx].drop(columns='price_bin')
#     val_df = df.loc[val_idx].drop(columns='price_bin')

# # Save to CSV
# train_df.to_csv('train_full.csv', index=False)
# val_df.to_csv('validation_set.csv', index=False)

# print("Stratified split complete:")
# print(f"Training set size: {len(train_df)}")
# print(f"Validation set size: {len(val_df)}")