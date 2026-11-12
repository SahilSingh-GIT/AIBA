# Telecom Customer Churn Timing Analysis Using Machine Learning: A Comparative Study of Survival Models

This project studies when telecom customers leave, not just whether they leave. It compares four models and checks which one gets the timing right.

## The idea

Most churn models answer "will this customer leave?" This project also asks "in which month does the risk show up?" A customer who has not left yet is still counted, so the data is treated as a timeline.

The models are:

- **Level model:** each factor changes the risk by the same amount every month.
- **Cox model:** a standard survival model with the same factors.
- **Boosted model:** a gradient-boosting model that can learn any pattern over time.
- **Screened model:** the new model. It picks only the factor-and-month pairs that matter, using one half of the customers, then checks them on the other half.

## Data

The IBM Telco Customer Churn sample, in `data/telco_customers.csv`. It has 7,043 customers. 11 with missing charges are removed, leaving 7,032.

## Main findings

- Monthly charge in the first month is the most stable timing effect. It was picked in all five random splits.
- The effect holds on the held-out customers.
- The screened model predicts the first month more accurately than the level and Cox models.
- Contract renewal timing, a common idea, does not show an effect in this data.

## Run it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_study.py
```

The run takes about a minute. Results and charts are saved in `results/`.

## Paper

The write-up is in `paper/paper.tex`. The compiled version is `paper/paper.pdf`.
