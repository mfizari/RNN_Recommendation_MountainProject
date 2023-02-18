# RNN_Recommendation_MountainProject

### Background

MountainProject is an online database of climbing areas and routes. Routes and areas are submitted by users and each page for a route contains descriptive information such as the route’s name, difficulty, length, type (trad, sport, boulder, etc), description, and location, as well as a section for users to comment on the route. Users can make an account and log the climbs they’ve done on a given day into their “ticks”. Users can also rate the quality of climbs on a 5-star scale. <br/>

There is currently no route recommendation system employed on MountainProject. While MountainProject does show the highest-rated routes (“classics”) for a given area in the area’s subpage, users have to navigate to that page to see those routes, which might make it difficult for users to see routes in areas that they don’t know about. 
To address this gap, I set out to build a recommendation engine for MountainProject routes. Recommendation systems are classically designed using content-based or collaborative filtering factorization methods. Content-based systems only recommend items that are like items a user has liked in the past. This means that the system may not be able to recommend new items that a user has not tried before. To overcome this, collaborative filtering is typically used. However classical matrix factorization for collaborative filtering suffers from the rating sparsity issue and is restricted to linear interactions. <br/>

I decided to use a recurrent neural network (RNN) approach for the basis of this recommendation system to address these issues. Deep learning has been recently shown to outperform classical methods in recommendation systems due to its ability to capture highly nonlinear, hidden interactions, its flexibility in feedback usage, and the ability to handle sparse data and contextual information ([1](https://static1.squarespace.com/static/59d9b2749f8dce3ebe4e676d/t/5dfaa6e33b6e8710da687359/1576707817521/JamesLe-Independent-Study-Report-Recommendation-Systems.pdf), [2](https://doogkong.github.io/2017/papers/paper2.pdf)). I chose RNNs due to their high performance in sequence prediction (e.g., time series data analysis, NLP). <br/>


#### Data

The dataset was scraped from MountainProject in a previous [project](https://github.com/mfizari/MountainProject-Web-Scraper). The entire tick profile of each user was scraped, where each tick contains various descriptive variables of the route, as well as personalized notes, the date, and how the user did on the climb. The content in the formatted scraped data is shown below, for a single user (transposed for clarity): <br/>


<img src="https://github.com/mfizari/RNN_Recommendation_MountainProject/blob/main/Data/raw_example.png" width=40% height=40%>

For each user, a list of ticks contains the following information for each tick:<br/>
`route_id`: A unique id assigned to each route by MountainProject (found in the route’s URL).<br/>
`ndays`: A scaled number of days that have passed since the tick date.<br/>
`rating_yds`: The YDS rating (difficulty of roped rock climbing) for a route. ‘None’ if the route is not a rock climb. <br/>
`rating_vscale` : The bouldering rating for a route (only applies to boulders that are typically climbed without a rope). ‘None’ if the route is not a boulder climb.<br/>
`rating_misc`: All other ratings (aid ratings, danger ratings, ice ratings, snow ratings, etc). ‘None’ if N/A.<br/>
`routetype`: The type of the route (Trad, Sport, Alpine, Snow, Mixed, Ice, etc). <br/>

#### Pre-processing
The goal of the RNN model is, for a list of `route_id`’s of length `n`, predict the values of the `n`th element in the sequence given the preceeding `n-1` elements. 
NNs require numeric inputs, so non-numeric data had to be tokenized in pre-processing. This included tokenization of `route_id`, all of the rating features, and `routetype`.<br/>

`rating_yds`:  All `rating_yds` values between 5.0 and 5.6 were grouped together under the value `Beginner` before tokenizing, to minimize the dictionary size. Slash, +-, and letter grades were removed, so that the possible values of `rating_yds` were `3rd, 4th, Easy 5th, Beginner, 5.7, 5.8` etc. <br/>

`rating_vscale` and `rating_misc`: A similar thing was done with these features, removing all slash grades and consolidating the possible options, especially due to the diversity in `rating_misc` (over 700 unique ratings).<br/>

After this, the rating features, `rotuetype`, and `route_id` were tokenized and converted to sequences. Additionally, `ndays` was binned in bin sizes of 15 days. Finally, since the RNN model is doing sequence-to-sequence prediction, the target sequence of `route_id` was produced by one-shifting. An individual sample in the full pre-processed dataset looked like this:<br/>




<br/>

#### Model<br/>
To avoid the vanishing gradient problem that can occur when dealing with long sequence, LSTM layers should be used in an RNN model. However, training with LSTMs can be very computationally expensive. Here, I decided to use NVIDIA CUDA Deep Neural Network library (cuDNN), which contains an extremely optimized LSTM layer implementation for GPU-accelerated training (`tf.compat.v1.keras.layers.CuDNNLSTM` in TensorFlow). However, this implementation does not support masking, which together with padding would allow variable sequence lengths. As a result, I started off by choosing a fixed sequence length (`max_sen_len`) of 50 ticks and restricted the dataset to only include the last 50 ticks of users with at least 50 ticks in their profile. This number was chosen to minimize the amount of user profiles not included in the dataset while maximizing the time-span of which ticks are considered. <br/>

To asses the importance of the various features, 3 different models were trained on 90% of the training dataset, with 10% left for validation. These models included 3 sets of features: `route_id` only, `route_id` and `ndays`, and all features listed above. Training was done on Paperspace Gradient notebooks. <br/>


#### Evaluation<br/>
To asses the performance of the various models, the average accuracy and precision were calculated for the multi-class classification problem on the validation set (the predicted class was selected by maximizing the probability). The results of the different models, compared to various intuitive baselines for route recommendation, are shown below:<br/>

<img src="https://github.com/mfizari/RNN_Recommendation_MountainProject/blob/main/Data/EvalMetrics.png" width=45% height=45%><br/>

Overall, the RNN models vastly outperform the baselines. Interestingly, adding the `ndays` feature improved the final metrics compared to only using `route_id` as a feature, but the addition of all the other features only marginally improved the performance when `ndays` was already included. Addition of `ndays` also significantly speed up the convergence time of the model during training (~100 epochs vs ~350 epochs). To minimize dataset sizes and processing times, it seems reasonable to only use these two features and still get high performance for inferences. <br/>


#### Future work
Since masking is not allowed in the cuDNN implementation of LSTM layers, we must use a fixed sequence length for inferences. This means that we cannot make predictions for users with fewer ticks that `max_sen_len`. To overcome this, we could use our currently trained model to extrpolate "past" ticks to fill in the sequence. Additionally, since we expect there to be significant seasonality in tick logs, we might want to adjust the truncation method to capture `route_id` sequences that span a longer time period, or use a larger value of `max_sen_len`. The later could be accomplished again by extrpolating past ticks for users with our current model. 



