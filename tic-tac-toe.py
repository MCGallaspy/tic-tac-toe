"""
A naive-bayes tic-tac-toe player. Given a game and a player to move as ('X' or 'O'), chooses the next move based on
which one has the maximum probability of winning. The model used is this: each of the nine positions on the board
is considered to be an IID random variable with three parameters: x - the probability that the position has an 'X' token
on it, o, the probability that the position has an 'O' token on it, and n, the probability that neither token is on the
position.

There are two independent parameters, since x + o + n = 1, so we only consider x and o for any position.

The positions are numbered like this:

1 | 2 | 3
---------
4 | 5 | 6
---------
7 | 8 | 9

Finally, we consider another variable W, which can take four values: player 'X' has won, player 'O' has won,
the game is unfinished, and the game is a draw. We'll consider two different possibilities for W: one in which
W is a deterministic function of the P_n's, and another in which it's a random variable with two independent parameters
that are some to-be-determined function of P_n.

The idea is first to see if a naive-bayes player can do well with a deterministic W model, and then compare the
performance to the probabilistic W model.
"""