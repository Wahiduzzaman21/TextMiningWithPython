# Run the script in following sequence for data preparation

1. DataExtraction for all category (change category from config.csv after completion of each category) - Will extract category file from all the content of Prothom Alo
2. UniqueTagExtractor for all category (change category from config.csv after completion of each category)
3. VocubularyExtraction - Preparing data for countvectorizer and tfidf transformer model (for each category and classification purpose)
4. CountVectorizerModel - Learn and create model for countvectorizer
5. TfIdfTransformerModel - Learn and create tfidf transformer model
6. PrepareCleanDataForMultiLabelClassification for all category (change category from config.csv after completion of each category) - Prepare clean for to fit for Multilabel classification model
7. Run Multilabel classification model for each category indivually and indivual pickle file will be created

