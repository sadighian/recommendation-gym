# Recommendation Gym for MovieLens
**Under construction as of 1-Aug-2019**
## 1. Overview

### 1.1 Summary
The purpose of this project was to experiment with the application
of deep reinforcement learning to recommendation systems.
More specifically, this project applies `Stable-Baselines`
algorithms to the `MovieLens 100k` data set.

To that end, the goal of the agent is to predict what rating 
a `user` will give to a given `movie`. 

### 1.2 Implementation

#### State
The observation space is comprised of data from the `MovieLens` data set:
- **user_mean:** Average rating given by a specific user_id
- **movie_mean:** Average rating for a specific movie_id
- **movie_genre_bucket:** One-hot of the movie type
- **age_bucket:** One-hot of user's age range
- **occupation_bucket:** One-hot of the user's job
- **gender_bucket:** One-hot of the user's gender (only M or F)

#### Reward
The reward structure is derived from the distance between the 
Agent's predicted rating vs. `user`'s actual rating.


## 2. Project Structure
```
gym_recommendation/
    data/           ...MovieLens 100k data set
    envs/
    tests/
    utils.py
experimeny.py
requirements.txt
setup.py
```

## 3. Getting Started
To be added

## 4. Appendix
To be added