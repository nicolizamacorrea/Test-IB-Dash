#!/usr/bin/env python
# coding: utf-8

# In[1]:


col_names=['IB_avg',
'FE_SAWDUST_RATIO',
'FE_THIRD_COMPRESION',
'FE_PRESSURE_14_15']

col_names1=[
'FE_SAWDUST_RATIO',
'FE_THIRD_COMPRESION',
'FE_PRESSURE_14_15']
# In[2]:


import pandas as pd
import numpy as np
import math
import plotly
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

df = pd.read_csv('exam1.csv', sep=',')

#df1 = pd.DataFrame(df_0, columns=col_names)
df1 = df[col_names]

# We change the most important features ranges to make them look like actual figures


df2 = df1.fillna(method='ffill').astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(df2.drop("IB_avg", axis=1), df2["IB_avg"], test_size=0.33, random_state=7)


# In[3]:


#model = Ridge(alpha=0.01)
#model = xgb.XGBRegressor()
model = RandomForestRegressor()

model.fit(X_train, y_train)

df_feature_importances = pd.DataFrame(model.feature_importances_*100,columns=["Importance"],index=col_names1)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)


# In[6]:




fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)

slider_1_label = df_feature_importances.index[0]
slider_1_min = int(math.floor( df2[slider_1_label].min()))
slider_1_mean = round( df2[slider_1_label].mean())
slider_1_max = int(round(df2[slider_1_label].max()))

slider_2_label = df_feature_importances.index[1]
slider_2_min = int(math.floor( df2[slider_2_label].min()))
slider_2_mean = round( df2[slider_2_label].mean())
slider_2_max = int(round( df2[slider_2_label].max()))

slider_3_label = df_feature_importances.index[2]
slider_3_min = int(math.floor( df2[slider_3_label].min()))
slider_3_mean = round( df2[slider_3_label].mean())
slider_3_max = int(round( df2[slider_3_label].max()))


# In[7]:




app = dash.Dash()

app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        
                        html.H1(children="Simulation TEST"),
                        
                        
                        dcc.Graph(figure=fig_features_importance),
                        
                        
                        html.H4(children=slider_1_label),

                        
                        dcc.Slider(
                            id='X1_slider',
                            min=slider_1_min,
                            max=slider_1_max,
                            step=0.5,
                            value=slider_1_mean,
                            marks={i: '{} '.format(i) for i in range(slider_1_min, slider_1_max+1)}
                            ),

                       
                        html.H4(children=slider_2_label),

                        dcc.Slider(
                            id='X2_slider',
                            min=slider_2_min,
                            max=slider_2_max,
                            step=0.5,
                            value=slider_2_mean,
                            marks={i: '{} '.format(i) for i in range(slider_2_min, slider_2_max+1)}
                        ),

                        html.H4(children=slider_3_label),

                        dcc.Slider(
                            id='X3_slider',
                            min=slider_3_min,
                            max=slider_3_max,
                            step=0.1,
                            value=slider_3_mean,
                            marks={i: '{}'.format(i) for i in np.linspace(slider_3_min, slider_3_max,1+(slider_3_max-slider_3_min)*5)},
                        ),
                        
                        
                        html.H2(id="prediction_result"),

                    ])


# In[ ]:



@app.callback(Output(component_id="prediction_result",component_property="children"),

              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value")])


def update_prediction(X1, X2, X3):

    
    input_X = np.array([X1,
                       X2,
                       X3
                      ]).reshape(1,-1)        
    
    
    prediction = model.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Prediccion: {}".format(round(prediction,2))

if __name__ == "__main__":
    #app.run_server()
    app.run_server(debug=True, threaded=True)
    
    


# In[ ]:




