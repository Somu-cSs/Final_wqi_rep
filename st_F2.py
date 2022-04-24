import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

ra=st.sidebar.radio('Select an option',['Analysis','Prediction'])
if ra == "Analysis":
    st.title('Water Quality Analysis')
    st.image("water.jpg")

    #    "Analysis"
    rad = st.sidebar.selectbox('Select', ['Vellore','Andhra Pradesh'])

    # This is for Ap data
    if rad == 'Andhra Pradesh':
        st.subheader("Andhra Pradesh Water Analysis")
        l = st.sidebar.radio('Select an option', ['January','February','March','April','May','June','July','August','September','October','November','December'])

        def month(l):
            url = 'https://raw.githubusercontent.com/Somu-Gen/WQI_Streamlit/main/'+l+'_wqi_2018.csv'
            return url

        url = month(l)
        df = pd.read_csv(url,encoding= 'unicode_escape',index_col=0)
        if st.checkbox('View Dataset'):
            st.dataframe(df)

        # Scatter Plot
        fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
        st.plotly_chart(fig1)

        #bar plot
        fig2 = px.bar(df, x="District/RO", y="wqi", color="quality",
                         color_discrete_map={
                             "Very poor": "red",
                             "Excellent": "#17becf",
                             "Good": "#72B7B2",
                             "Not suitable": '#AF0038',
                             "Poor": "#A777F1"},
                         title="Explicit color mapping")
            # Set the visibility ON
        fig2.update_xaxes(title='District/RO', visible=True, showticklabels=True)
            # Set the visibility OFF
        fig2.update_yaxes(title='WQI', visible=True, showticklabels=False)
            #st.plotly_chart(fig2)


            #Pie
            # pull is given as a fraction of the pie radius
        fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi)])
            #st.plotly_chart(fig3)

        columns = st.columns((2,1))
        with columns[0]:
            st.plotly_chart(fig2)


        with columns[1]:
            st.plotly_chart(fig3)

        # using Plotly SunBurst
        import plotly.express as px
        import numpy as np

        fig_s = px.sunburst(df, path=['District/RO', 'Station_name'], values='wqi',
                            color='wqi', hover_data=['quality'],
                            color_continuous_scale='bluyl',
                            )
        st.plotly_chart(fig_s, use_container_width=True)

        # Correlation
        if st.checkbox('Correlation plot'):
            plt.figure(figsize=(13, 8))
            sns.heatmap(df.drop(["S.No."], axis=1).corr(), annot=True, cmap='terrain')
            st.pyplot(plt)

        figl = px.line(df, x='TDS', y="wqi", title="TDS vs WQI plot")
        #st.plotly_chart(figl)

        figm = px.line(df, x='Na', y="wqi", title="Na vs WQI plot")
        #st.plotly_chart(figm)

        fign = px.line(df, x='Cl', y="wqi", title="Cl vs WQI plot")
        #st.plotly_chart(fign)

        figo = px.line(df, x='K', y="wqi", title="K vs WQI plot")
        #st.plotly_chart(figo)

        columns = st.columns((1, 1))
        with columns[0]:
            st.plotly_chart(figl)

        with columns[1]:
            st.plotly_chart(figm)

        columns = st.columns((1, 1))
        with columns[0]:
            st.plotly_chart(fign)

        with columns[1]:
            st.plotly_chart(figo)


        # Voilin plot
        if st.checkbox('Statistical Analysis'):
            fig = px.violin(df, x="District/RO", y="wqi")
            st.plotly_chart(fig)
            # st.write(df)


    if rad == 'Vellore':
        st.title("Vellore water quality analysis")
        df =pd.read_csv("https://raw.githubusercontent.com/Somu-Gen/WQI_Streamlit/main/vellore_Dy.csv")
        if st.checkbox("View Dataset"):
            st.write(df)
        # Scatter Plot
        fig1 = px.scatter(df, x="Date", y="wqi", color="quality", color_discrete_map={
            "Very poor": "red",
            "Excellent": "#17becf",
            "Good": "#72B7B2",
            "Not suitable": '#AF0038',
            "Poor": "#A777F1"},
                          title="<b>Vellore DATE VS WQI</b> ")
        #st.plotly_chart(fig1)
        fig1_1 = px.scatter(df, x="Village", y="wqi", color="quality", color_discrete_map={
            "Very poor": "red",
            "Excellent": "#17becf",
            "Good": "#72B7B2",
            "Not suitable": '#AF0038',
            "Poor": "#A777F1"},
                          title="<b>Vellore VILLAGE vs WQI</b> ")
        #st.plotly_chart(fig1_1)
        columns = st.columns((2, 1))
        with columns[0]:
            st.plotly_chart(fig1)

        with columns[1]:
            st.plotly_chart(fig1_1)

        # bar plot
        fig2 = px.bar(df, x="Date", y="wqi", color="quality",
                      color_discrete_map={
                          "Very poor": "red",
                          "Excellent": "#17becf",
                          "Good": "#72B7B2",
                          "Not suitable": '#AF0038',
                          "Poor": "#A777F1"},
                      title="Explicit color mapping")
        # Set the visibility ON
        fig2.update_xaxes(title='Date', visible=True, showticklabels=True)
        # Set the visibility OFF
        fig2.update_yaxes(title='WQI', visible=True, showticklabels=False)
        #st.plotly_chart(fig2)

        # Pie
        # pull is given as a fraction of the pie radius
        fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi)])
        #st.plotly_chart(fig3)
        columns = st.columns((1, 1))
        with columns[0]:
            st.plotly_chart(fig2)

        with columns[1]:
            st.plotly_chart(fig3)
        #OLS plot
        st.subheader('Ordinary least-squares')
        st.image('OLS.png')
        # Correlation
        if st.checkbox('Correlation plot'):
            plt.figure(figsize=(13, 8))
            sns.heatmap(df.drop(["sno"], axis=1).corr(), annot=True, cmap='terrain')
            st.pyplot(plt)

        # VIOLIN PLOT
        if st.checkbox('Statistical Analysis'):
            fig = px.violin(df, x="Date", y="wqi")
            st.plotly_chart(fig)
            # st.write(df)

        # Line plot
        figl = px.line(df, x='Date', y="wqi", title="Date vs WQI plot")
        # st.plotly_chart(figl)

        figm = px.line(df, x='F', y="wqi", title="F vs WQI plot")
        st.plotly_chart(figm)

        fign = px.line(df, x='Date', y="F", title="F vs Date plot")
        # st.plotly_chart(fign)

        columns = st.columns((1, 1))
        with columns[0]:
            st.plotly_chart(figl)

        with columns[1]:
            st.plotly_chart(fign)

        # From here its year wise analysis
        st.title('Year wise Analysis')
        k = st.selectbox('Select a year',['1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'] )

        dfy = df[['Village','Date', 'wqi', 'quality', 'year']]
        def year_fun(y):
            df1 = dfy[dfy['year'] == int(y)]
            return df1

        df2=year_fun(k)

        # Scatter Plot
        fig1 = px.scatter(df2, x="Date", y="wqi", color="quality", color_discrete_map={
            "Very poor": "red",
            "Excellent": "#17becf",
            "Good": "#72B7B2",
            "Not suitable": '#AF0038',
            "Poor": "#A777F1"},
                          title="<b>WATER QUALITY INDEX</b> ")
        st.plotly_chart(fig1)

        # bar plot
        fig2 = px.bar(df2, x="Date", y="wqi", color="quality",
                      color_discrete_map={
                          "Very poor": "red",
                          "Excellent": "#17becf",
                          "Good": "#72B7B2",
                          "Not suitable": '#AF0038',
                          "Poor": "#A777F1"},
                      title="Explicit color mapping")
        # Set the visibility ON
        fig2.update_xaxes(title='Date', visible=True, showticklabels=True)
        # Set the visibility OFF
        fig2.update_yaxes(title='WQI', visible=True, showticklabels=False)
        # st.plotly_chart(fig2)

        # Pie
        # pull is given as a fraction of the pie radius
        fig3 = go.Figure(data=[go.Pie(labels=df2.quality, values=df2.wqi)])
        # st.plotly_chart(fig3)

        columns = st.columns((1, 1))
        with columns[0]:
            st.plotly_chart(fig2)

        with columns[1]:
            st.plotly_chart(fig3)


        # using Plotly SunBurst
        import plotly.express as px
        import numpy as np

        fig_s = px.sunburst(df2, path=['Date','Village'], values='wqi',
                            color='wqi', hover_data=['quality'],
                            color_continuous_scale='bluyl',
                            )
        st.plotly_chart(fig_s, use_container_width=True)


# Prediction
if ra == "Prediction":
    st.title('Water Quality Prediction')
    #st.image("prediction.jpg")

    # input data
    n_in1 = st.sidebar.number_input("Enter the Total Dissolved Solids:")
    n_in2 = st.sidebar.number_input("Enter the Nitrates :")
    n_in3 = st.sidebar.number_input("Enter the Calcium :")
    n_in4 = st.sidebar.number_input("Enter the Magnesium :")
    n_in5 = st.sidebar.number_input("Enter the Sodium :")
    n_in6 = st.sidebar.number_input("Enter the Potassium :")
    n_in7 = st.sidebar.number_input("Enter the Chlorine:")
    n_in8 = st.sidebar.number_input("Enter the Sulphate :")
    n_in9 = st.sidebar.number_input("Enter the F :")
    n_in10 = st.sidebar.number_input("Enter the pH :")

    # st.markdown(n_in1)
    st.markdown(
        f""""

         TDS :{n_in1}
         Nitrate :{n_in2}
         Ca :{n_in3}
         Mg :{n_in4}
         Na :{n_in5}
         K :{n_in6}
         Cl:{n_in7}
         SO4:{n_in8}
         F :{n_in9}
         pH : {n_in10}
    """
    )
    def load_keywords_fromfile():
    # reading the dictionnary des 15 keyword
        with open('ridge.pkl', 'rb') as handle: 
            data = handle.read() 
        # reconstructing the data as dictionary 
        lst_keywords_byclass = pickle.loads(data) 
        return lst_keywords_byclass
    lst_keywords_byclass = load_keywords_fromfile()
    #model = pickle.load(open("ridge.pkl", 'rb'))

    # input_d=(8.5,200,250,75,50,250,200,500,12,1)
    input_data = (n_in1, n_in2, n_in3, n_in4, n_in5, n_in6, n_in7, n_in8, n_in9, n_in10)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = lst_keywords_byclass.predict(input_data_reshaped)
    # st.write('Water Quality Index',prediction)
    ca = prediction.astype(np.float)
    st.title('WQI:')
    st.subheader(ca)
    if 0< prediction <= 25:
        st.subheader('Quality status: Excellent')
    elif 25 < prediction <= 50:
        st.subheader('Quality status: Good')
        st.image("Good.png")
    elif 50 < prediction <= 75:
        st.subheader('Quality status: Poor')
    elif 75 < prediction <= 100:
        st.subheader('Quality status: Very poor')
    elif prediction > 100:
        st.subheader('Quality status: Not suitable')
    # st.wrte(input_data)
