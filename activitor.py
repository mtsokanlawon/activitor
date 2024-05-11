import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder
import pyarrow as pa

# the test set ??
data_path = r"C:\Users\Owner\Downloads\deploy_data.xls"
test_set = pd.read_csv(data_path)

# Image links
im1 = r"C:\Users\Owner\Downloads\Walking_Phone_Accelerometer_Series.png"
im2 = r"C:\Users\Owner\Downloads\Walking_Phone_Gyroscope_Series.png"
im3 = r"C:\Users\Owner\Downloads\Walking_Watch_Accelerometer_Series.png"
im4 = r"C:\Users\Owner\Downloads\Walking_Watch_Gyroscope_Series.png"
im5 = r"C:\Users\Owner\Downloads\Jogging_Phone_Accelerometer_Series.png"
im6 = r"C:\Users\Owner\Downloads\Jogging_Phone_Gyroscope_Series.png"
im7 = r"C:\Users\Owner\Downloads\Jogging_Watch_Accelerometer_Series.png"
im8 = r"C:\Users\Owner\Downloads\Jogging_Watch_Gyroscope_Series.png"
im9 = r"C:\Users\Owner\Downloads\Pasta_Phone_Accelerometer_Series.png"
im10 = r"C:\Users\Owner\Downloads\Pasta_Phone_Gyroscope_Series.png"
im11 = r"C:\Users\Owner\Downloads\Pasta_Watch_Accelerometer_Series.png"
im12 = r"C:\Users\Owner\Downloads\Pasta_Watch_Gyroscope_Series.png"
im13 = r"C:\Users\Owner\Downloads\Typing_Phone_Accelerometer_Series.png"
im14 = r"C:\Users\Owner\Downloads\Typing_Phone_Gyroscope_Series.png"
im15 = r"C:\Users\Owner\Downloads\Typing_Watch_Accelerometer_Series.png"
im16 = r"C:\Users\Owner\Downloads\Typing_Watch_Gyroscope_Series.png"

phone_accel = [im1, im5, im9, im13]
phone_gyro = [im2, im6, im10, im14]
watch_accel = [im3, im7, im11, im15]
watch_gyro = [im4, im8, im12, im16]

walking_p = [im1, im2]
walking_w = [ im3, im4]

jogging_p = [im5, im6,]
jogging_w =[ im7, im8]

pasta_p = [im8, im10]
pasta_w =[im11, im12]

typing_w = [im13, im14]
typing_p = [im15, im16]

model_path = r"C:\Users\Owner\Downloads\classification_model (1).pkl"
with open(model_path, 'rb') as model_file:
    classifier = pickle.load(model_file)

def classify_activity(dataX):
    prediction = classifier.predict(dataX)

    # Create a dictionary for Transforming the encoded labels to Activity
    ActivityDict = {     
                        0: "Walking", 
                        1:"Jogging",2:"Stairs",
                        3:"Sitting", 4:"Standing",
                        5:'Typing', 6:"Brushing Teeth",
                        7:"Eating Soup", 8:'Eating Chips',9:'Eating Pasta',
                        10:'Drinking from Cup', 11:'Eating Sandwich',
                        12:'kicking (Soccer Ball)', 13:'Playing Catch w/Tennis Ball',
                        14:'Dribbling (Basketball)', 15:'Writing',
                        16:'Clapping', 17:'Folding Clothes'}
    
    # since thee prediction is an inordinate number, we could access the value by the key in Dictionary. 
    return f"The activity recognised is {ActivityDict[prediction[0]]}."

def preprocess(df):
    df.set_index(df.columns[0], inplace=True)
    # keep column names
    column_names = list(df.columns)
    column_names.remove('"ACTIVITY"')
    column_names.remove('"class"')
    #print(np.array(column_names).size)
    #column_names
    
    df.drop(columns='"ACTIVITY"', inplace=True)
    df.drop(columns='"class"', inplace=True)
    data_all = df.values
    X_data = data_all.astype(None)
    transfmd_df = pd.DataFrame(X_data)
    transfmd_df.columns = column_names

    transfmd_df.reset_index(drop=True, inplace=True)

    return transfmd_df



def main():
 
    st.set_page_config(layout='centered') 

    st.title("Human Activity Recognitor")

    html_temp = """
    <div style="background-color:rgb(255, 75, 75);padding:0px">
    <h2 style="color:white;text-align:center;">Phone and Smartwatch Data Activity Recognition</h2>
    </div>
    """
    #st.markdown(html_temp, unsafe_allow_html=True)

    html_temp_line = """
    <div style="background-color:black;padding:0px">
    <h2 style="color:white;text-align:center;"></h2>
    </div>
    """

    home_tab, tab2 = st.tabs(["Home", "Visualizations", ])

    #home_tab.markdown(html_temp, unsafe_allow_html=True)
    if home_tab:
        home_tab.markdown(html_temp_line, unsafe_allow_html=True)
        home_tab.subheader("View Data")
        home_tab.write("You may want to view the table of activity code to identify activities in the test set")
    if home_tab.button("View Activity Code"):
            home_tab.write("Table of Activity Code")
            home_tab.table({"ACTIVITY":"CODE",
                        "Walking":"A", "Jogging":"B", "Stairs":"C",
                        "Sitting":"D", "Standing":'E',
                        'Typing':'F', "Brushing Teeth":"G",
                        "Eating Soup":'H', 'Eating Chips':'I','Eating Pasta':'J',
                        'Drinking from Cup':'K', 'Eating Sandwich':'L',
                        'kicking (Soccer Ball)':'M', 'Playing Catch w/Tennis Ball':'O',
                        'Dribbling (Basketball)':'P', 'Writing':'Q',
                        'Clapping':'R', 'Folding Clothes':'S'})
            
    if home_tab.button("View Test Set"):
        home_tab.dataframe(test_set.sample(60))
        
        #direct user to sidebar
        home_tab.success("Enter Index of Test data from The Sidebar")

    if tab2:
        tab2.subheader("Time Series of Readings")
        tab2.write("Viewing time series of readings from phone(accelerometer, gyroscope) and watch(accelerometer, gyroscope) for selected activities.")
        tab2.write("Each image visualizes 180 rows of data, signifying approximately 10seconds of data reading across all axes.")

        # Visualize Data from Phone
        tab2.subheader("Phone")
        if tab2.button("Phone Accelerometer"):
            tab2.image(phone_accel)
        if tab2.button("Phone Gyroscope"):
            tab2.image(phone_gyro)

        # Visualize Data from Watch
        tab2.subheader("Watch")
        if tab2.button("watch Accelerometer"):
            tab2.image(watch_accel)
        if tab2.button("Watch Gyroscope"):
            tab2.image(watch_gyro)

       # Visualize Activity Data Based on Device
        tab2.subheader("Compare Device-Meter")
        if tab2.button("Walking"):
            tab2.write("Phone")
            tab2.image(walking_p)
            tab2.write("Watch")
            tab2.image(walking_w)

        if tab2.button("jogging"):
            tab2.write("Phone")
            tab2.image(jogging_p)
            tab2.write("Watch")
            tab2.image(jogging_w)

        if tab2.button("Pasta"):
            tab2.write("Phone")
            tab2.image(pasta_p)
            tab2.write("Watch")
            tab2.image(pasta_w)

        if tab2.button("Typing"):
            tab2.write("Phone")
            tab2.image(typing_p)
            tab2.write("Watch")
            tab2.image(typing_w)

    
    # Get data row input from user
    st.sidebar.subheader("Data Input")
    data_index = st.sidebar.number_input("Enter a test data index:", min_value=0, max_value=(test_set.shape[0]-1))
        
    selected_data = test_set.loc[[data_index]]
    
    # display selection as df
    if data_index:
        home_tab.subheader("Selected Data")
        home_tab.dataframe(selected_data)
        home_tab.success("Selection Successful! Processing Input...")

        processed = preprocess(selected_data)
        home_tab.success("Pre-Processing Completed!")
        home_tab.dataframe(processed)
        
        result = ""
        if home_tab.button("Recognize Activity"):
            result = classify_activity(processed)
            home_tab.success(result)


if __name__ == "__main__":
    main()                       
