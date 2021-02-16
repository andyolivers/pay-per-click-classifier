# Main Problem Definition
You are in the Data Science team at FlixBus. Your colleagues from the marketing department (specifically from the **PPC** (**P**ay **P**er **C**lick) team) came to you with a problem.

The PPC team handles the PPC ads, which are: for example, when you search for something on Google, and then some advertised results show up before the *organic* ones. For these ads FlixBus pays only when someone clicks on the ad, hence the name.

To help them manage such ads, Google provides them with daily data about the performance of their ads. This data includes (all on a daily basis):
* **clicks**: the number of clicks on the ad.
* **impressions**: the number of times the ad showed up (whether it was clicked or not).
* **cost**: the total cost that we paid (to Google) for the ad. (in microcents: 1e-6 €)
* **conversions**: the number of purchases. (fractions are normal and expected)
* **conversion_value**: the total revenue (in €)
* **ad_position**: the median position of the ad when it showed up
* **ad_location**: a derived variable we calculate describing where the ad showed up on the search results page (the top 3 positions show up on the top of the page, and the rest are displayed on the side of the page) (also a median)
 
The problem is: Google is soon deprecating the "ad position" parameters and is not going to provide them. The PPC team depends on these parameters to understand competition, market behaviour, and our ads' performance.
So they were wondering if we could use the rest of the performance metrics to predict the ad position; so we can continue to use it after Google deprecates it.

They provided you with the daily performance data of a specific ad for about 13 months.

# Tasks
1. Build a solution to predict the ad position.
2. (Optional bonus) Which ad position is most efficient for us? Where efficiency is defined as **revenue/cost**. (Don't forget to mention how much that efficiency is)
 

# Evaluation
Don't spend too much time on enhancing the prediction results. If the prediction results are unsatisfactory, you can just leave a note on why you think this is the case, and what you would try in order to enhance them (without actually needing to do that). The quality of your prediction is of course still a factor for the evaluation, but also how you evaluate your model, your approach of tackling the problem, your data handling, **the reproducibility of your results**, and the quality of your solution in general.

# How to submit
Our favourite approach is a Python Jupyter Notebook, but feel free to use another approach if you're more comfortable with it.
Make sure your solution can be understood and run on our side, as well as your results demonstrated.
Compress your solution into an archive and send it back using the link you were sent in the email.
(Optional: You may also send your solution as a git repo, but **please do not publish it publicly**.)