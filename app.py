import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
import wikipedia as wp
import requests
from dataclasses import dataclass
from sklearn.metrics import accuracy_score 
import openpyxl
from PIL import Image

# ========================= Input Feature Info =========================
labels = [ # used in sidebar
  "Ammonia as N (MG/L)",
  "Acid Neutralizing Capacity (UEQ/L)",
  "Calcium (MG/L)",
  "Chloride (MG/L)",
  "Color (ALPHA PT-CO)",
  "Conductivity (US/CM at 25C)",
  "Dissolved Oxygen Content (MG/L)",
  "Magnesium (MG/L)",
  "Nitrite + Nitrate as N (MG N/L)",
  "Total Nitrogen (MG/L)",
  "Potassium (MG/L)",
  "Phosphorus (UG/L)",
  "Silica (MG/L)",
  "Sodium (MG/L)",
  "Sulfate (MG/L)",
  "Turbidity (NTU)",
]

@dataclass
class Input_Feature():
    dataset_label: str # label used by machine
    mean_value: float
    max_value: float

features = []
df_stats = pd.read_csv("app_contents/df2_stats.csv") # read straight from dataset (or at least statistics of dataset) for procedural values
for i in range(len(labels)):
    dataset_label = df_stats.columns[i + 1]
    mean_value = float(df_stats.iat[1, i + 1])
    max_value = round(float(df_stats.iat[7, i + 1]))
    max_value = round(max_value, -(len(str(max_value)) - 1))
    features.append(Input_Feature(dataset_label, mean_value, max_value))

# ========================= SIDEBAR =========================
# ------ Sidebar Info ------
st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar
st.sidebar.write("Click on the '?' button for any chemistry feature for a brief description of what they are.")
st.sidebar.write("Sliders are based on a logarithmic scale.")

# ------ Sidebar Sliders/Use Input ------
selected_element = 0 # represents which water characteristic selected for Info Box
def display_feature(label, default, max, id): # represents a single box containing a slider, text input, reset button, and ? button
    global selected_element
    with st.expander("**" + label + "**", expanded = True):
        # ------ Slider ------
        # While pH did not have to be normalized, the other water characteristics tend to fall under an uneven
        # distribution such that some values are abnormally large. Thus, normalize them on slider for more precision
        # at lower selection values
        slider_default = pow(default, 0.25) / pow(max, 0.25) # log normalize
        def slider_change():
            # have number input match slider value if changed (must convert from normalized values)
            st.session_state["input_" + str(id)] = pow(st.session_state["slider_" + str(id)] * pow(max, 0.25), 4) 
        st.slider(
                    label = label,
                    value = slider_default,
                    min_value = 0.0,
                    max_value = 1.0,
                    format="",
                    label_visibility = "collapsed",
                    key = "slider_" + str(id),
                    on_change = slider_change
        )

        col1, col2, col3 = st.columns([4, 1.25, 1])
        with col1:
            # ------ Number Input ------
            def input_change():
                # have slider match number input if changed (must convert to normalized values)
                st.session_state["slider_" + str(id)] = pow(st.session_state["input_" + str(id)], 0.25) / pow(max, 0.25)
            st.number_input(
                        label = label,
                        value = default,
                        min_value = 0.0,
                        format = "%.6f",
                        label_visibility = "collapsed",
                        key = "input_" + str(id),
                        on_change = input_change
            )
        with col2:
            # ------ Reset Button ------
            def reset():
                st.session_state["slider_" + str(id)] = slider_default # set to normalized equivalent
                st.session_state["input_" + str(id)] = default # set to default value number input was initialized as 
            st.button("Reset", key = "button_" + str(id), on_click = reset)
        with col3:
            # ------ Info Button ------
            if st.button("?", key = "info_" + str(id)):
                selected_element = id
        
        return st.session_state["input_" + str(id)]

# ------ Reset All Buttons ------
def reset_all():
    for i in range(len(labels)):
        # same thing as individual reset buttons, just loop through all
        default = features[i].mean_value
        max = features[i].max_value
        slider_default = pow(default, 0.25) / pow(max, 0.25)
        st.session_state["slider_" + str(i)] = slider_default
        st.session_state["input_" + str(i)] = default

# ------ Displaying Sidebar Elements ------
user_input = []
with st.sidebar:
    st.button("Reset All", on_click = reset_all, key = "reset_0")
    for i in range(len(labels)):
        # get user input for features
        user_input.append(display_feature(labels[i], features[i].mean_value, features[i].max_value, i))
    st.button("Reset All", on_click = reset_all, key = "reset_1")


def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.wine_type = df.wine_type.map({'white':0, 'red':1})
    return df

# ========================= DESCRIPTION =========================
st.header("pH Prediction Using Water Chemistry")
st.image(Image.open("pictures/water_image.jpg"), caption='A scientific view of water',use_column_width=True)
st.write("This app predicts the **pH** of a body of water given its chemical features (such as total nitrogen, conductivity, etc.) as input. In order to use, simply enter the chemical features on the sidebar. Results will be displayed and updated automatically below. It is important to note that the default values are not the values of a typical body of water (if one can even classify such a thing), but merely the mean value of the datapoints used to train this model.")
st.write("In a realistic scenario, someone who has the ability to measure the chemical components of water will also probably have the ability to measure the pH as well. Thus, this app mostly exists as a proof-of-concept, as hypothetically water chemistry could be used to predict other water characteristics such as organic density.")

# ========================= INFO BOX =========================
# ------ General Info ------
@dataclass
class General_Info():
    label: str
    description: str
    picture: Image
    caption: str
# information can be added into info.txt following the format:
#   Characteristic Name
#   Description
#   Image File Name (images retrieved from pictures subdirectory)
#   Caption
#   (newline)
# due to lack of time haven't filled these out
info_file = open("app_contents/info.txt", "r") 
@st.cache
def getGeneralInfo():
    infos = []
    for i in range(len(labels)):
        label = info_file.readline().rstrip()
        description = info_file.readline().rstrip()
        image = Image.open("pictures/" + info_file.readline().rstrip())
        caption = info_file.readline().rstrip()
        infos.append(General_Info(label, description, image, caption))
        info_file.readline()
    return infos
infos = getGeneralInfo()

# ------ Wikipedia Info ------
@dataclass
class Wiki_Info():
    query: str
    description: str
    picture: Image
    caption: str

wiki_file = open("app_contents/wikipedia_queries.txt", "r")
@st.cache
def getWikiInfo():
    wikis = []
    for i in range(len(labels)):
        query = wiki_file.readline().rstrip()
        # by default act as if query failed, because most likely it did. not really sure why
        description = "Error accessing Wikipedia page."
        picture = Image.open("pictures/missing_image.jpg")
        caption = "No image found"
        try:
            wiki_page = wp.page(query)
            description = wiki_page.summary
            image_url = wiki_page.images[0]
            picture = Image.open(requests.get(image_url, stream=True).raw)
            caption = "Retrieved from Wikipedia"
        except:
            pass
        wikis.append(Wiki_Info(query, description, picture, caption))
    return wikis
wikis = getWikiInfo()

with st.expander("**Feature Information**", expanded = True):
    st.subheader(infos[selected_element].label)
    tab1, tab2 = st.tabs(["General Info", "Wikipedia Excerpt"])
    with tab1:
        # ------ Displaying General Info ------
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(infos[selected_element].description)
        with col2:
            st.image(infos[selected_element].picture, caption = infos[selected_element].caption, use_column_width = True)
    with tab2:
        # ------ Display Wikipedia Info ------
        # same format used for General Info
        col1, col2 = st.columns([2,1])
        with col1:
            st.write(wikis[selected_element].description)
        with col2:
            st.image(wikis[selected_element].picture, caption = wikis[selected_element].caption, use_column_width = True)

# ========================= MODEL PREDICTION =========================
st.write("Shown below is a graph representing the confidence level for each of the three possible pH range values, that being Acidic, Neutral, and Alkaline. Rarely will a machine model be 100\% certain about a prediction, and thus we deal in confidence levels instead of absolutes.")
st.write("As stated above, the graph and result below it is updated real-time. Appropriate to real-life, some chemical characteristics affect the pH more than others. In some cases, my model appears to reflect this--reducing the Acid Neutralizing Capacity will notably make things more acidic; however, I have noticed some inaccuracies, such as Ammonia--despite being basic--increasing acidity. Whether or not this is due to an inaccuracy in the model or maybe in real life Ammonia's affect on pH is not as straightforward, I am unsure.")
# load model from .joblib file
model = joblib.load(open("final_model.joblib","rb"))
# convert user input into a dataframe usable by machine model
user_input_dict = {features[i].dataset_label: user_input[i] for i in range(len(labels))}
converted_input = pd.DataFrame(user_input_dict, index=[0])

prediction = model.predict(converted_input)
prediction_proba = model.predict_proba(converted_input)

# ========================= PREDICTION RESULTS =========================
# each of the prediction probabilities aligns with one of the 3 pH ranges
prediction_values = prediction_proba[0]*100
prediction_labels = ["Acidic", "Neutral", "Alkaline"]
prediction_dict = {prediction_labels[i] : prediction_values[i] for i in range(3)}

def display_confidence_level(values, labels):
    # plot a simple bar graph
    fig, ax = plt.subplots()
    ax.bar(labels, values, width = 0.5)
    ax.set_ylim(ymin=0, ymax=100)
    ax.set_title('Prediction Confidence Level ', loc='center', weight='bold')
    ax.set_xlabel("Water pH Range", size=9)
    ax.set_ylabel("Percentage Confident Level", size=9)
    st.pyplot(fig)
display_confidence_level(prediction_values, prediction_labels)

result = max(prediction_dict, key=prediction_dict.get)
clarifier = ""
if prediction_dict[result] <= 80:
    clarifier = "(probably)"
if prediction_dict[result] <= 60:
    clarifier = "(maybe)"

st.subheader("The water is: **" + result + "** " + clarifier)

# ========================= USER FILE INPUT =========================
# ------ Display Format ------
st.write("This web app not only allows you to make predictions based on the input on the sidebars, but through your own uploaded data in .csv or .xlsx format--This will allow you to make multiple predictions in bulk automatically. Your data must be a .csv file formatted akin to the table shown below.")
st.write(converted_input)
with st.expander("Same Data in Dictionary Form", expanded = True):
    st.write(user_input_dict)
st.write("You do not have to provide all input features. Any columns that do not match the formatting for any input features will be dropped. Do not replace missing columns with 0s--rather, upload your .csv file with the missing columns and this web app will replace it with default values. Considering it is the input feature, you also do not need to provide a PH column. If you do, however, alongside the model's predicted pH values an accuracy score will be displayed representing how my model did compared to your own (presumably accurate) input.")
# ------ Remapping Function ------
st.write("For extra usability, I've added a way to map the column names on your file to those used by my machine model so can leave the dataset untouched and not have to modify it yourself. Simply type in your dataset's corresponding column name below. For example, if your dataset has a column named \"NH3\" instead of \"AMMONIA\" (case matters), type \"NH3\" in the space by \"AMMONIA\". Spaces left blank will not have any renaming done.")
st.write("The field for \"PH RANGE\" only exists if your dataset happens to map pH the same way I died, which was by classifying values in the ranges of 0-6, 6-8, and 8-14 as the numerical values 0, 1, and 2 respectively. This web app automatically does this if provided raw pH values (any columns named or remapped to \"PH RANGE\" will be ignored).")
with st.expander("Remapping Column Names", expanded = True):
    new_names = []
    mapping_dict = {}
    for i in range(len(labels)):
        mapping_dict[features[i].dataset_label] = ""
    mapping_dict["PH"] = ""
    mapping_dict["PH RANGE"] = ""

    for key in mapping_dict:
        col1, col2 = st.columns([1, 2.5])
        with col1:
            st.write(key)
        with col2:
            mapping_dict[key] = st.text_input(
                                    label = key,
                                    key = "remap_" + key,
                                    label_visibility = "collapsed",
                                    placeholder = "Corresponding name for " + key + "..."
                                )
# ------ Upload User File ------
user_file = st.file_uploader("Upload Your Own Dataset Here")
user_csv = None

#@st.cache(suppress_st_warning=True)
def getUserCSV():
    # convert file to pandas dataframe
    if user_file.name.split(".")[-1] == "csv":
        user_csv = pd.read_csv(user_file, encoding= 'unicode_escape')
    if user_file.name.split(".")[-1] == "xlsx":
        user_csv = pd.read_excel(user_file)
    return user_csv
if user_file is not None:
    user_csv = getUserCSV()
    if (user_csv is None):
        st.write("Unsupported file type--Must be either .csv or .xlsx")
    else:
        st.write("Your data:")
        st.write(user_csv)

# ========================= USER FILE PROCESSING =========================
def processUserCSV(user_csv):
    total_columns = len(user_csv.columns)
    total_rows = len(user_csv.index)
    processed_flag = 0
    # ------ Processing Nulls ------
    nulls = user_csv.isna().sum().sum()
    user_csv = user_csv.apply(pd.to_numeric, errors="coerce") # remove strings and replace with null
    # replace all null with mean values
    values = {}
    for feat in user_csv.columns:
        if user_csv[feat].isna().any():
            values[feat] = round(user_csv[feat].mean(),2)
        user_csv.fillna(value=values,inplace=True)
    # ------ Remap Columns Names ------
    renamed = 0
    for key in mapping_dict:
        if mapping_dict[key] == "" or mapping_dict[key] is None or mapping_dict[key] == key:
            continue
        try:
            user_csv.rename(columns={mapping_dict[key] : key}, inplace= True)
            renamed += 1
        except:
            pass
    # ------ Drop Unecessary Columns ------
    # pretty sure this part is done automatically by the model.fit() function but whatever
    dropped = 0
    for column in user_csv.columns:
        # automatically convert "PH" to "PH RANGE", overriding it in the process if it already exists
        if column == "PH":
            bins = [0,6,8,14] 
            bin_labels = [0, 1, 2]
            user_csv["PH RANGE"]= pd.cut(x=user_csv["PH"], bins=bins, labels = bin_labels)
            processed_flag = 1
        # drop any columns not features in dataset
        if column not in df_stats.drop("Unnamed: 0", axis = 1).columns and column != "PH RANGE":
            user_csv.drop(column, axis = 1, inplace=True)
            dropped += 1
    # ------ Add Missing Columns ------
    added = 0
    for i in range(len(labels)):
        if features[i].dataset_label not in user_csv.columns:
            user_csv[features[i].dataset_label] = features[i].mean_value
            added += 1

    if (processed_flag + renamed + dropped + added >= 1):
        # sort added columns
        user_csv = user_csv.reindex(sorted(user_csv.columns), axis=1)
        if (processed_flag == 1):
            ph_row = user_csv["PH RANGE"]
            user_csv.drop(columns=["PH RANGE"], inplace = True)
            user_csv.insert(0, "PH RANGE", ph_row)

        st.write("Your data (processed):")
        st.write(user_csv)
        
        # give user some info about stuff
        def processed_captions(number, total, label):
            if number > 0:
                st.caption(str(number) + "\/" + str(total) + label)
        processed_captions(nulls, total_rows * total_columns, " null datapoints replaced")
        processed_captions(renamed, total_columns, " columns renamed")
        processed_captions(dropped, total_columns, " columns dropped")
        processed_captions(added, len(labels), " columns added")
        if (processed_flag == 1):
            st.caption("\"PH\" column processed and renamed to \"PH RANGE\"")

    X = user_csv.copy()
    y = None
    if ("PH RANGE" in X.columns):
        y = X["PH RANGE"]
        X.drop(columns=["PH RANGE"], inplace = True)
        
    
    predictions = model.predict(X)
    output_csv = X.copy()
    output_csv.insert(0, "PH RANGE", predictions)
    st.subheader("Predicted pH Values")
    st.write(output_csv) 

    # ------ Downloading Results ------
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    st.download_button("Download Output .csv File", convert_df(output_csv), "output.csv")
    st.download_button("Download Predicted pH Column Only", convert_df(output_csv.pop("PH RANGE")), "predicted_ph.csv")
    if processed_flag == 1:
        st.download_button("Download Processed Input .csv", convert_df(user_csv), user_file.name.split(".csv")[0] + "_processed.csv")

# ========================= USER INPUT ACCURACY SCORING =========================
    if y is not None:
        score = accuracy_score(y, predictions)
        st.subheader("Model accuracy score: " + str(round(score, 6)))
if user_csv is not None:
    processUserCSV(user_csv)