# Source
https://amazon-reviews-2023.github.io/

# Description
Uses the Amazon Fashion category above the link, and extracts users who interacted with more than 3 items and items that were interacted with by more than 2 users.

80% of data is used to train(i.e., graph_train.csv) and 20% of data is used to test(i.e., graph_test.csv).

Each data is composed of 4 columns: user_id,item_id,rating,sentiment.

The _sentiment_ column means emotional score of users' item review(The more positive review, the closer to 1).