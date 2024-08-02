#imports
import dowhy.datasets
import dowhy
from dowhy import CausalModel
import json
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import logging
import logging.config

# Config dict to suppress logging of dowhy
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)

#use the linear dataset generator to create a dataset
#only run this the first time around and save the dictionary output as below
#there is no random_seed parameter in the linear_dataset function so the dataset will be different each time
data_dict = dowhy.datasets.linear_dataset(
    beta=10,
    num_common_causes=1,
    num_effect_modifiers=2,
    num_samples=10000,
    outcome_is_binary=True,
    treatment_is_binary=True,
)

#function to save the data as we don't have a random seed argument in linear_dataset()
def save_data(data_dict, output_dir='output'):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract DataFrame and save to CSV
    df = data_dict.pop('df')
    df.to_csv(f"{output_dir}/oci_intro_blog_generic_data.csv", index=False)
    
    # Save the rest of the dictionary to JSON
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(data_dict, f, indent=4)

#function to load the data
def load_data(input_dir='output'):
    # Load DataFrame from CSV
    df = pd.read_csv(f"{input_dir}/oci_intro_blog_generic_data.csv")
    
    # Load metadata from JSON
    with open(f"{input_dir}/metadata.json", 'r') as f:
        data_dict = json.load(f)
    
    # Add DataFrame back to the dictionary
    data_dict['df'] = df
    
    return data_dict

# Save the data
#save_data(data_dict)

# Load the data
loaded_data = load_data()

#get pandas dataframe from df dictionary
data = loaded_data["df"]

#check the head
data.head()

#in v0 and y replace True with 1 and False with 0 for clarity
#only need to run this the first time
data["v0"] = data["v0"].astype(int)
data["y"] = data["y"].astype(int)

#print the true Average Treatment Effect (ATE)
print("The true Average Treatment Effect (ATE) of the treatment on the outcome is:", loaded_data["ate"])

#calculate the difference in the share of 1s in y between the two groups
diff_by_v0 = data.groupby("v0")["y"].mean().diff().iloc[-1]
print(diff_by_v0)

#difference in diff_by_v0 and loaded_data['ate]
print("The difference between the 'naive' effect and the true Average Treatment Effect", round(diff_by_v0 - loaded_data["ate"],4))

#Initialize the causal model
model=CausalModel(
        data = data,
        treatment=loaded_data["treatment_name"],
        outcome=loaded_data["outcome_name"],
        graph=loaded_data["gml_graph"]
        )

#visuall inspect the graphical causal model
model.view_model()

#Identify the estimand
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Estimate the causal effect using logistic regression
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.generalized_linear_model",
    method_params={
        "glm_family": sm.families.Binomial(),
        "predict_score": True
    },
    confidence_intervals=True,
    test_significance=True
)

# Print the results
print(estimate)

# Refute the causal effect
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter", n_jobs=-1)
print(refute_results)

# Refute the causal effect
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause", n_jobs=-1)
print(refute_results)

# Refute the causal effect
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.8, n_jobs=-1)
print(refute_results)

