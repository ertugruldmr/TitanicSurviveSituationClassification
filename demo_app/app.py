import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn

# File Paths
model_path = 'rf_model.sav'
endoing_path = "cat_encods.json"
component_config_path = "component_configs.json"

# predefined

cat_cols = ['sex', 'embarked', 'class', 'who']
num_cols = ['age', 'sibsp', 'parch', 'fare', 'alone']

target = "survived"

feature_order = ['sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'alone']

# Loading the files
model = pickle.load(open(model_path, 'rb'))

# loading the classes & type casting the encoding indexes
classes = json.load(open(endoing_path, "r"))
classes = {k:{int(num):cat for num,cat in v.items() } for k,v in classes.items()}

inverse_class = {col:{val:key for key, val in clss.items()}  for col, clss in classes.items()}

labels = ["not survived", "survived"]#classes[target].values()

feature_limitations = json.load(open(component_config_path, "r"))

# Example Cases
examples = [
    ['female', 24.0, 2.5, 2.0, 5.5759491, 'S', 'First', 'woman', 0.0],
    ['male', 2.5, 0.0, 2.0, 3.637696679934592, 'C', 'Second', 'child',0.0],
    ['female', 29, 2.5, 2.0, 4.256321678298823, 'S','Third', 'woman', 0.0],
    ['male', 32, 0.0, 0.0, 2.169053700369523, 'Q', 'Third', 'man',1.0]
]


# Util Functions
def decode(col, data):
  return classes[col][data]

def encode(col, str_data):
  return inverse_class[col][str_data]

def feature_decode(df):

  # exclude the target var
  cat_cols = list(classes.keys())
  if "survived" in cat_cols:
    cat_cols.remove("survived")

  for col in cat_cols:
     df[col] = decode(col, df[col])

  return df

def feature_encode(df):
  
  # exclude the target var
  cat_cols = list(classes.keys())
  if "survived" in cat_cols:
    cat_cols.remove("survived")
  
  for col in cat_cols:
     df[col] = encode(col, df[col])
  
  return df

def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  features = feature_encode(features)
  features = np.array(features).reshape(-1,len(feature_order))

  # prediction
  probabilities = model.predict_proba(features) #.predict(features)
  probs = probabilities.flatten()

  # output form
  results = {l : np.round(p, 3) for l, p in zip(labels, probs)}

  return results

# creating the components
inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )

# creating the app
demo_app = gr.Interface(predict, inputs, "label", examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()