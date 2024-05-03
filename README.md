# Key Phrase Tracking

The `gen_control_limits()` function takes several inputs and generates multiple outputs. Let's break down how the function works step by step:

## 1. Input Parameters

The function accepts various input parameters, including:

- `df`: A DataFrame containing the data.
- `date_col`: The name of the column in the DataFrame that contains the dates.
- `start_date`: The start date for filtering the data.
- `end_date`: The end date for filtering the data.
- `cur_date`: The current date for analysis.
- `stopwords`: A list of stopwords to be excluded from the analysis.
- `stopgrams`: A list of stopgrams (phrases) to be excluded from the analysis.
- `ngram_len`: The length of n-grams to be generated.
- `ngram_min_freq`: The minimum frequency threshold for n-grams.
- `ngram_cutoff_periods`: The number of periods to use as a cutoff for n-grams.
- `ngram_pval_thresh`: The p-value threshold for the chi-squared test.
- `ngram_pmi_thresh`: The threshold for pointwise mutual information (PMI) scores.
- `max_tfidf_ngrams`: The maximum number of n-grams to be generated using TF-IDF.
- `rule1_topn`: The number of top n-grams to be considered for Rule 1.
- `d2`: A constant value used in calculating control limits.

## 2. Data Preprocessing

The function performs several data preprocessing steps:

- Formatting dates: The function converts the date column in the DataFrame to a standardized format.
- Filtering data: The function subsets the DataFrame based on the specified date range.
- Handling missing values: The function replaces any missing values in the `stopwords` and `stopgrams` parameters with empty lists.

## 3. N-gram Generation

The function generates n-grams using three different scoring methods: PMI, chi-squared, and t-test. For each scoring method, the function performs the following steps:

- Extracting word tokens: The function extracts the word tokens from the filtered data.
- Calculating scores: The function calculates scores (PMI, chi-squared, or t-test) for each n-gram based on its frequency and other statistical measures.
- Filtering n-grams: The function filters out n-grams based on the specified thresholds for PMI, p-values, and minimum frequency.
- Combining n-grams: The function combines the n-grams generated from all three scoring methods and removes any n-grams that contain stopgrams.

## 4. N-gram Substitution

The function substitutes static n-grams into the text by replacing them with their corresponding phrases. This step helps incorporate static phrases into the document.

## 5. Count Frequency and Percentage Calculation

The function calculates count frequencies for each n-gram and the percentage of total frequencies for each unigram. It builds a dictionary of count statistics for each date.

## 6. Control Limit Calculation

The function calculates control limits based on the count statistics. It performs the following steps:

- Determining the start and end dates for training: The function identifies the start and end dates for building control limits based on the specified date range.
- Calculating mean and moving average: The function calculates the mean and moving average for each n-gram based on the training period.
- Calculating standard deviation: The function calculates the standard deviation based on the moving average and a constant value (`d2`).
- Calculating upper and lower control limits: The function calculates the upper and lower control limits based on the mean and standard deviation.
- Calculating zone limits: The function calculates upper and lower limits for different zones (Zone A, Zone B, and Zone C) based on the mean and standard deviation.

## 7. Rule Generation

The function generates rules to identify key phrases that are emerging. It generates five rules:

- Rule 1: Beyond limits - identifies n-grams that have one or more points beyond the control limits.
- Rule 2: Zone A - identifies n-grams that have two out of three consecutive points in Zone A or beyond.
- Rule 3: Zone B - identifies n-grams that have four out of five consecutive points in Zone B or beyond.
- Rule 4: Zone C - identifies n-grams that have seven or more consecutive points on one side of the average (in Zone C or beyond).
- Rule 5: Trending up - identifies n-grams that have seven consecutive points trending up.

## 8. Output

The function returns multiple outputs, including:

- `DataDev`: A DataFrame containing the count statistics and control limits for each n-gram.
Overall, the `gen_control_limits()` function takes the input parameters, preprocesses the data, generates ngrams, filters and substitutes ngrams, calculates count frequencies and control limits, generates rules, and returns the transformed DataFrame and rule subsets as outputs.