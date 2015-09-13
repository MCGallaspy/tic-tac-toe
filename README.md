What is this
------------

It's my way of learning Bayesian modelling and how to use the pymc3 library by way of example.
The goal is to create an agent that can play tic-tac-toe perfectly.
Some of the comments/docstrings in the various files may not be up-to-date, or may reflect my
incorrect understanding when I wrote them.

Right now, I'm modelling the game as a naive Bayes multinomial classifier. The game has a certain
state represented by the multinomial random variable W (who is winning? has values X, O, or Neither).
Each playable position in the game is also considered a multinomial random variable (with states X, O,
or Neither again representing which player has moved there). The hyperparameters (theta and beta, 
respectively) of each multinomial are considered to be shared among all instances. In reality, 
we consider three distinct betas for the whole model -- one for each value of W. We consider each
beta to be conditionally independent given W.

Thus the random variable W does not directly influence the distribution for each of its nine T variables, 
but it does select which of the three betas govern the hyperparameters of the Ts.


How to run this
---------------

Get ipython notebook and pymc3. I use anaconda, since the dependendecies are complex. Steps:

#. Get anaconda
#. Create a new conda environment with some deps. See [Theano installation help](http://deeplearning.net/software/theano/install_windows.html#alternative-anaconda)
  in particular. Also see the pymc3 README for other deps.
#. Once you have an environment with all dependencies, clone pymc3 repo, then run `python setup.py install`.
#. In your environment run `conda install jupyter` to get the notebook viewer.
#. `jupyter notebook` to launch the notebook viewer.
#. Open and run `tic-tac-toe.ipynb`. It's not fast. Enjoy.