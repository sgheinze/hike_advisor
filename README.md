# HikeAdvisor

<img src="/figures/hiker_sil.png" alt='blah' width="48" />

Welcome to the GitHub repo for HikeAdvisor, a recommendation system for hikes in California. HikeAdvisor was designed to help new and casual hikers, who often spend hours reading trail descriptions and reviews on websites like AllTrails.com, choose their next hike. The product utilizes hike data from online websites to make personalized hike recommendations to users based on previously enjoyed hikes and desired attributes in future hikes. 

Please visit www.hikeadvisor.net to get your recommendations.

----
## Why I built HikeAdvisor
HikeAdvisor was built while I was attending the Insight Data Science program, a post-doctoral training fellowship designed to help recently graduated Ph.D.s transition from the world of academia to data science. Having completed my doctoral degree in California, I had recently developed a casual interest in hiking. One of the things that always frustrated me about hiking was the amount of time it took to scope out new hikes. I often spent hours on AllTrails.com reading through user reviews of hikes trying to decide whether or not I would enjoy a hike, or even if a hike was appropriate for my skill level. I always thought that there should be a better way about choosing new hikes, so I decided to try to tackle that problem at Insight.

----
## Usage
Using HikeAdvisor is simple and can be divided into two steps:
1. In the upper box, select which hikes you've enjoyed in the past. Multiple or no hikes can be selected.
2. In the bottom boxes, select the desired attributes you want in your next hike. Multiple attributes or none can be selected.

----
## How it works
HikeAdvisor utilizes hybrid recommendation system that incorporates both collaborative filtering and content-based information. The collaborative filtering method makes automatic predictions about the interests of a user (i.e. which hikes they might like) by utilizing the collective interests of many users. For instance, if many users like hike "A" and hike "B," it's highly likely that a new user who likes hike "A" will also like hike "B." The content-based information describes the attributes of each hike, e.g. the difficulty, distance, elevation, etc. The content-based information helps the collaborative filtering method decide which hikes are similar.

When a user interacts with HikeAdvisor, the recommendation system uses *all* of the inputted information in order to make recommendations. The inputs are converted into vectorized embeddings in the latent space and the list of returned hikes is built using item-item (cosine) similarity. 

