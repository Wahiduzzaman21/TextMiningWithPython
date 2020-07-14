# Run the script in following sequence
1. DataExtraction for all category (change category from config.csv after completion of each category) - Will extract category file from all the content of Prothom Alo
2. UniqueTagExtractor for all category (change category from config.csv after completion of each category)
3. DataPreparationForWordEmbedding - Preparing data for countvectorizer and tfidf transformer model
4. CountVectorizerModel - Learn and create model for countvectorizer
5. TfIdfTransformerModel - Learn and create tfidf transformer model
6. PrepareCleanDataForMultiLabelClassification for all category (change category from config.csv after completion of each category) - Prepare clean for to fit for Multilabel classification model