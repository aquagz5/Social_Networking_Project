## Yelp Friendship Recommendations
Bryan Michalek | Tony Quagliata
### Goals and Project Direction
The goal of this project is to develop a recommendation system for enhancing user engagement on Yelp's platform by suggesting potential friendships between users based on their review behaviors and existing social connections. We will frame this as a link prediction problem, where the goal is to recommend the top ~5-10 users that each user should be friends with, utilizing graph neural network methods. By leveraging the current graph structure along with node features such as types of businesses reviewed, activity level of a user, membership duration, user type, and engagement metrics, the project aims to accurately predict and recommend friendships that are likely to be accepted, thereby expanding users' social networks on the platform.

### Benefit to Yelp
This project is important for Yelp's product as it will enhance user engagement and foster community connections within the platform. If we can facilitate meaningful connections between users through friend recommendations, Yelp can potentially increase user satisfaction, retention, and overall activity on the platform.

### Datasets Used
We are using a Yelp dataset, which comprises a list of users and their Yelp friend connections. This makes up our graph. There are additional tables that will be useful for constructing our own node features including reviews left by users and descriptions of businesses.

### Timeline
* Week 0 (March 25-31): Dataset selection, graph creation (we must derive the edge list and features ourself).
* Week 1 (April 1-7): Basic graph analysis (feature EDA, summarize graph statistics), write introduction of research paper, create intro presentation slides, begin writing GNN code (if time allows).
* Week 2 (April 8-14): Write GNN code, train and evaluate the model, report friend recommendations for each user, write methods, results, and conclusion sections of paper, create presentation slides for methods, results, conclusion.
* Week 3 (April 15-21): Write abstract of paper, create executive summary presentation slide, project cleanup, presentation practice, submission of research paper / slides / code documentation.
