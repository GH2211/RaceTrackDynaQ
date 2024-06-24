# RaceTrackDynaQ
Reinforcement Learning for a race car

<img src="Images/track.png" alt="alt text" width="700" height="400">

Implementing the DynaQ algorithm to improve on a Q Learning agent.

<img src="Images/ResultsGraph.png" alt="alt text" width="700" height="400">

The graph above shows that the modifications resulted in quicker learning and better final performance. The quickened performance can be attributed to the “planning”, bootstrapping information for the model allowing it to learn quicker. The long-term performance improvements are a combination of the DynaQ algorithm as well as the epsilon decay discouraging exploration as we approach the optima and learner decay reducing overfitting to noise. 


# References

Racetrack environment code by Dr Joshua Evans

Plotting code and Q-Learning plot (correct_returns_q.json) by Dr Joshua Evans 
