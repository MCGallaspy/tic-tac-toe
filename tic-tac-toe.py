"""
We'll model the game of tic-tac-toe using a multinomial naive bayes model.
The graph is given in the image plate_model.png.
Explained here:
Winner and Tokens are both multinomial distributions with hyperparameters Theta and Beta_i, respectively.
W is the set of values for Winner = {X wins, O wins, Neither wins}
T is the set of values of Tokens = {X, O, Empty}
Theta is a Dirichlet distribution.
Each of the Beta_i is a Dirichlet distribution -- we assume they are conditionally indepedent given Winner.
P is the the set of positions -- we number each position in the tic-tac-toe board from top to bottom, left to right
  starting with 1, for a total of 9 positions.
G is the training set of games.

The model has two steps -- in the first step, we learn the parameters Theta and Beta_i given a training set G.
This is done using Bayesian parameter estimation techniques.
Specifically we generate a distribution P(Theta, Beta_i | G), which we can factor into:

  P(Theta | G) * [product over i in W of P(Beta_i | G)]

And determine each posterior distribution individually. Let W_g be the observed values of Winner in G, and
  similarly let T_g be the observed values of each Token in G. Then:

  P(Theta | G) = P(Theta) * P(W_g | Theta) / integral of numerator dTheta

P(Theta) is the prior -- a Dirichlet distribution. Use an initial value, but then the posterior from the last run of the
  algorithm becomes the prior for the next.
P(W_g | Theta) is the likelihood function -- Winner is a multinomial distribution, and given Theta we can easily
   compute the likelihood.
The denominator is a normalizing constant.
For each Beta_i:

  P(Beta_i | G ) = P(Beta_i ) * P(T_g | Beta_i) / intergal of numerator dBeta_i

First, for each game, the Beta_i is chosen to correspond to the observed value of W_g for the game.
The observed value of W_g dictates which Beta_i is being learned.
P(Beta_i) is again a Dirichlet prior. Again the posterior of the last run becomes the prior of the next.
P(T_g | Beta_i) is again a likelihood function -- Token is a multinomial with hyperparameters Beta_i.

Once posteriors are calculated, we move on to step 2 of the model. We consider the hyperparameters as observed and
  calculate P(Winner = w | T_new), where w is some state in W, and T_new is a set of observed tokens.

  P(W=w | T_new) = P(W=w) * P(T_new | W=w) / P(T_new)

Given the hyperparameters as observed, each term on the RHS can be calculated. P(W=w) is easily determined from
  P(Theta | G). Similarly, P(T_new | W=w) is a likelihood function again given the P(Beta_i | G) where Beta_i is
  the posterior corresponding to w. The denominator is again a normalizing constant:

  P(T_new) = integral of P(T_new | W=w) * P(W=w) dW = sum over W of P(T_new | W=w=i) * P(W=w_i)

The integral becomes a sum of the likelihoods of the positions given a winner w_i in W.

Given a state of the board and a player, P, to move, we perform this calculation once for each empty position on the
  board. The state w is fixed and chosen so that P is the winner. Each of T_new is the old state but with one of the
  empty positions filled with P's token. We choose to make the move that maximizes the probability.
"""
import numpy as np
from pymc3 import Model, Dirichlet, find_MAP, NUTS, sample, Multinomial, traceplot
from scipy import optimize

model = Model()
X, O, Neither = "X", "O", "Neither"
WINNER = (X, O, Neither,)
TOKENS = (X, O, Neither,)

# prior observations. Mappings of variables, W, T_1, T_2, ..., T_9
games = [
    {
      "W": Neither,
      "T1": X,
      "T2": O,
      "T3": X,
      "T4": X,
      "T5": O,
      "T6": O,
      "T7": O,
      "T8": X,
      "T9": X,
    },
    {
      "W": O,
      "T1": O,
      "T2": O,
      "T3": Neither,
      "T4": O,
      "T5": X,
      "T6": X,
      "T7": O,
      "T8": X,
      "T9": Neither,
    },
]

def w_ind(s):
    """
    Maps a symbol in WINNER to an index, for consistency
    :param s: A symbol in WINNER
    :return: an integer index
    """
    return WINNER.index(s)

def t_ind(t):
    """
    Maps a symbol in TOKENS to an index, for consistency
    :param t: A symbol in TOKENS
    :return: an integer index
    """
    return TOKENS.index(t)

with model:
    theta = Dirichlet("theta", np.ones(3), shape=3)
    beta = {t: Dirichlet(name="beta_%s" % t, a=np.ones(3), shape=3) for t in WINNER}

    for g_ind, game in enumerate(games):
        wobs = [0, 0, 0]
        wobs[w_ind(game["W"])] += 1
        w = Multinomial("w_%s" % g_ind, 1, theta, observed=wobs)
        for pos in range(1, 10):
            tobs = [0, 0, 0]
            tobs[t_ind(game["T%s" % pos])] += 1
            t = Multinomial("t_g%s_%s" % (g_ind, pos), 1, beta[game["W"]], observed=tobs)

    print("finding map estimate...")
    map_estimate = find_MAP(fmin=optimize.fmin_powell)

    print("map estimate:")
    print(map_estimate)

    print("NUTS")
    step = NUTS(scaling=map_estimate)

    print("sample")
    hierarchical_trace = sample(2000, step, start=map_estimate)

traceplot(hierarchical_trace[500:])
