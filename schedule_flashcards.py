import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns


'''
TODO: Major bug: currently just predicts whichever class is in the majority
100% of time, doesn't distinguish based on features interval might not be
helpful, I don't often just attempt random very long intervals
-- maybe survivor bias is distorting the dataset, my reviews are not a random
enough selection
-- maybe targeting success at 85-90% makes it hard to get data on what causes
failures
(note: currently undersampling positives, removing good data to get closer to
50/50, problem persists, was 0 false negatives now 0 false positives, algo
never gambles on a theory and refuses to learn)

just looking at the curve of interval to success rate was illustrative, longer
intervals gradually decline in success, until about 100 days
there is a linear relationship there to exploit, but still positive
maybe need to undersample / oversample so that this curve crosses the 50/50
line? Currently goes from ~90 down to ~70,
maybe need to undersample positives to 55 pos/45 neg ratio so its rewards are
actually affected by this relationship ****

big picture goals:
if 40 reviews today, choose 40 with highest impact
which reviews should i do today? (over some impact threshold)
impact score = greatest increase of % chance of recalling a random card in
the deck tomorrow relative to no review
or, which card will most reduce the number of reviews over time to hit some
recall probability for the entire deck
don't just review "right before you forget" -- review the cards that are the
most malleable and buy you the most time
don't review cards you know extremely well, don't review cards you will just
forget again tomorrow.

rate impact of review per card
predict likelihood of success per card
if not reviewed, likelihood of success tomorrow (next week, next month...)
if success, likelihood of success tomorrow, next week, etc.
if fail, likelihood of success tomorrow, next week, etc.
this is a graph, you want to minimize the area or rev/day at x% retention
(recall success)
algo needs to make intelligent choices on when to add new card vs when to
review old card
algo needs to seek valuable data by inserting random reviews to learn abt
failure rates, calculate value of the algo learning per review, have it conduct
intelligent experiments?

should approximate:
case 0: card has no reviews. p(success) - all cards in deck success first try /
  all cards tried, add one success one fail for laplace
case 1: long string of successes. avg interval weighted to most recent.
  increase ease, accelerate with exponential backoff
case 2: string of higher than average failures with shorter than avg intervals.
  re-sort until all unseen cards processed.
case 3: string of successes with recent failure -- relearning up to last
  successful interval more rapid, maybe blip from being tired or bad day and
  can return to last successful interval very quickly
case 4: string of successes recent persistent failures -- relearn slower,
  facing interference, bury if too slow
case 5: mix of successes and failures, reviewing impact on retention
  inconsistent to negligible. if can consistently hit x month interval like a
  ceiling, sustain at ease = 1 if x large enough, if celing is low or reviews
  just seem totally random, bury.

need:
predict chance of success right now
predict chance of success at t+1 if not reviewed
predict chance of success t+1 if reviewed and successful
predict chance of success t+1 if reviewed and failed
value of review = [P(T0) * P(T1|T0)] + [(¬P(T0)) * P(T1|¬T0)] - P(T1|¬RT0)
notation a bit awkward here, T0 means "reviewed at time 0 && was successful"
¬T0 means reviewed and unsuccessful, ¬RT0 means not reviewed
'''

''' ROADMAP
1. adjust sampling to be 52% p / 48% n to see if it can detect interval
  crossover
2. adjust hyperparameters
3. feature engineering
4. adversarial data collection? have it schedule 10% of reviews to learn more
  about performance
5. other tools, random forest? linear regression for some continuous variable?

3. features:
---features expected to be weaker signals but could be useful in some
   situations---
- collection success rate  # won't vary between cards, not informative?
- collection success rate by interval
- collection success rate by interval / longest successful interval
- card's full review history # (too complex and already captured?)
- deck success rate  # deck ids missing from test data, edit quantified self to
  pull deck ids?
- deck success rate at various intervals
- streaks, longest cumulative successes (is this redundant with longest ivl?)
- median latency to answer  # too noisy, sometimes you get distracted, uneven
-- distribution, can't answer in -1 seconds, can answer in an hour, dominated
-- by outliers
-- decks will vary a lot -- some have longer content, some are single words
--  ok MAYBE you could just ask if the median or 20%ile latency (cut most of
-- the long tail) is above or below the deck average, that might work
- prior 'hard' and 'easy' scores -- note that if i'm correctly predicting when
-- cards are hard and easy then anki will adjust review timing so that success
-- rate has a limited impact
- datetime format has unexpected tradeoffs - dt objects easier to work with in
-- general, but algorithms expect floats at some point, just confirm these are
-- in a rational format throughout
'''


def process_review_history(filename):
    '''Process the .csv export generated by Quantified Self add on into a
    pandas dataframe. The rows in this csv contain a span between reviews,
    so it lists review1 and review2, then the next row might be review2 and
    review3. Some processing is needed to ensure we capture each review,
    since most of them show up in the data twice, but the first and last
    reviews only show up once, and in separate columns.

    Tried a loop that mostly just grabbed the review2 column, only adding
    review1 when we first see a new card. It was somehow slower than just
    buiding two complete dataframes and merging them while dropping duplicates.
    '''
    df = pd.read_csv(filename)

    # create two dataframes from Date1, Answer1 and Date2, Answer2
    df_1 = df[['Date1', 'Answer1', 'Card ID']].copy()
    df_2 = df[['Date2', 'Answer2', 'Card ID']].copy()

    # rename columns to have same column names
    df_1.columns = ['Date', 'Answer', 'Card ID']
    df_2.columns = ['Date', 'Answer', 'Card ID']

    df_merged = pd.concat([df_1, df_2]).drop_duplicates()
    df_merged['Date'] = pd.to_datetime(df_merged['Date'])
    df_merged = df_merged.sort_values(['Card ID', 'Date'])
    df_merged = df_merged.reset_index(drop=True)

    return df_merged


def add_features(df):
    """Add additional features derived from collected data.
    """
    # Add Last date and interval
    df['Last Date'] = df.groupby('Card ID')['Date'].shift()
    df['Interval'] = df['Date'] - df['Last Date']

    # Add Last interval, Last result
    df['Last Interval'] = df.groupby('Card ID')['Interval'].shift()
    df['Last Answer'] = df.groupby('Card ID')['Answer'].shift()

    # Calculate the cumulative success rate.
    df['Successful'] = df['Answer'].apply(lambda x: 1 if x > 1 else 0)
    df['Cumulative Successes'] = df.groupby('Card ID')['Successful'].cumsum()
    df['Cumulative Reviews'] = df.groupby('Card ID').cumcount() + 1
    df['Cumulative Success Rate'] = (df['Cumulative Successes']
                                     / df['Cumulative Reviews'])

    # Add smoothed cumulative success rate
    # This might help minimize the issue of overconfidence at few reviews
    # TODO: tune additive smoothing values
    adsmooth_num = 0.85  # approx. collection overall success rate as prior
    adsmooth_den = 1
    df['Smoothed Success Rate'] = ((df['Cumulative Successes'] + adsmooth_num)
                                   / (df['Cumulative Reviews'] + adsmooth_den))

    # Add longest successful interval
    df['Longest Successful Interval'] = (
        df[df['Successful'] == True].groupby('Card ID')['Interval'].cummax()
    )

    df['Longest Successful Interval'] = (
        df.groupby('Card ID')['Longest Successful Interval']
        .fillna(method='ffill')
    )

    df['Longest Successful Interval'].fillna(pd.Timedelta(seconds=0),
                                             inplace=True)

    # Add shortest failed interval
    df['Shortest Failed Interval'] = (
        df[df['Successful'] == False].groupby('Card ID')['Interval'].cummin()
    )
    df['Shortest Failed Interval'] = (
        df.groupby('Card ID')['Shortest Failed Interval']
        .fillna(method='ffill')
    )

    return df


def normalize_dates(df):
    """Converts pd dates in a df dataframe to Unix epoch seconds"""
    df['Date'] = df['Date'].apply(lambda x: x.timestamp())
    df['Last Date'] = (
        df['Last Date']
        .apply(lambda x: x.timestamp() if pd.notnull(x) else x)
    )
    df['Interval'] = (
        df['Interval']
        .apply(lambda x: x.total_seconds() if pd.notnull(x) else x)
    )
    df['Last Interval'] = (
        df['Last Interval']
        .apply(lambda x: x.total_seconds() if pd.notnull(x) else x)
    )
    df['Longest Successful Interval'] = (
        df['Longest Successful Interval']
        .apply(lambda x: x.total_seconds() if pd.notnull(x) else x)
    )
    df['Shortest Failed Interval'] = (
        df['Shortest Failed Interval']
        .apply(lambda x: x.total_seconds() if pd.notnull(x) else x)
    )

    return df


def plot_interval_success(df):
    # plotting relationship between interval and success
    # interval bins are poorly labeled but are 1-2 days, 2-4 days, 4-8, etc

    # Convert 'Interval' from Timedelta to seconds
    df['Interval_seconds'] = df['Interval'].dt.total_seconds()

    # Then, create bins for 'Interval_seconds'
    bin_size = 60*60*24  # bin size of 1 day
    # bins = np.arange(0, df['Interval_seconds'].max() + bin_size, bin_size)

    bins = [0, (60*60*24)]
    while bins[-1] < df['Interval_seconds'].max():
        # Double the last value in the list and append it
        bins.append(bins[-1]*2)
    bins = np.array(bins)

    df['Interval_binned'] = pd.cut(df['Interval_seconds'], bins)

    # Then, group by 'Interval_binned' and calculate success rate
    success_rate = df.groupby('Interval_binned')['Successful'].mean()

    # Plot the success rate for each bin
    plt.figure(figsize=(10, 6))
    success_rate.plot(kind='line')
    plt.title('Success Rate vs Interval')
    plt.xlabel('Interval (days)')
    plt.ylabel('Success Rate')

    plt.show()


df = process_review_history('RevisionHistory.csv')
print('CSV read.')
df = add_features(df)
print('Features added.')
df = normalize_dates(df)
print('Dates normalized.')

'''
NaT errors:
Last Date: NaT if 1st review of card
Interval: NaT if 1st rev of card
Last Answer: NaT if 1st rev of card
Last Interval: NaT if no prior Interval, 1st or 2nd rev of card
Shortest Failed Interval: NaT until first fail
'''
# resolve NaT issues by removing first review of each card,
# removing the shortest failure column
# and imputing the last interval as this interval for second reviews
# cast Last Date and Interval to floats after NaTs removed
df = df.dropna(subset=['Last Date'])
df['Last Interval'].fillna(df['Interval'], inplace=True)
df.drop('Shortest Failed Interval', axis=1, inplace=True)

df['Last Date'] = df['Last Date'].astype(float)
df['Interval'] = df['Interval'].astype(float)

print("NaT issues resolved.")

# Force display of all rows and columns
# pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)
# print(df.head(50))


# Balance df
num_unsuccessful = df['Successful'].value_counts()[0]
successful_indices = df[df['Successful'] == 1].index
random_indices = np.random.choice(successful_indices,
                                  num_unsuccessful,
                                  replace=False)
unsuccessful_indices = df[df['Successful'] == 0].index
under_sample_indices = np.concatenate([unsuccessful_indices, random_indices])
df_balanced = df.loc[under_sample_indices]
print("Dataframe balanced.")


X = df_balanced.drop('Successful', axis=1)
y = df_balanced['Successful']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print("Training and testing sets split.")


# Define the model
# TODO: How many layers? How many neurons?
neural_model = Sequential()
# First hidden layer with 64 neurons
neural_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# Second hidden layer with 32 neurons
neural_model.add(Dense(32, activation='relu'))
# other layers
neural_model.add(Dense(32, activation='relu'))
neural_model.add(Dense(32, activation='relu'))
neural_model.add(Dense(32, activation='relu'))
neural_model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
neural_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

# Fit the model
neural_model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model
_, accuracy = neural_model.evaluate(X_test, y_test)
print('Neural Network Accuracy: %.2f' % (accuracy*100))


# Make predictions with your Keras model
y_pred_prob = neural_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Net (keras) Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict on the test set
logistic_predictions = logistic_model.predict(X_test)

# Check the accuracy
print("Logistic Regression Accuracy: ",
      accuracy_score(y_test, logistic_predictions))

from sklearn.metrics import confusion_matrix

# make predictions on your test data
y_pred = logistic_model.predict(X_test)

# generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression (sklearn) Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)

