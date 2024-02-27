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
    rad = st.sidebar.selectbox('Select a State', ['Andhra Pradesh', 'Vellore'])

    # This is for Ap data
    if rad == 'Andhra Pradesh':
        st.subheader("Andhra Pradesh Water Analysis")
        #year = st.sidebar.selectbox('Select a Year', ['2018', '2019'])
        l = st.sidebar.radio('Select an option', ['January','February','March','April','May','June','July','August','September','October','November','December'])
        if l =="January":
            # plotting the violin chart
            df = pd.read_csv("https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/January_wqi_2018.csv")
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            #Bar plot
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
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
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


           #Correlation
            if st.checkbox('Correlation plot'):
                plt.figure(figsize=(13, 8))
                sns.heatmap(df.drop(["S.No."], axis=1).corr(), annot=True, cmap='terrain')
                st.pyplot(plt)

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)

        if l == "February":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/February_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)
        if l == "March":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/March_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)
        if l =="April":
            # plotting the violin chart
            df = pd.read_csv("https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/April_wqi_2018.csv")
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            #Bar plot
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
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
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


           #Correlation
            if st.checkbox('Correlation plot'):
                plt.figure(figsize=(13, 8))
                sns.heatmap(df.drop(["S.No."], axis=1).corr(), annot=True, cmap='terrain')
                st.pyplot(plt)

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)

        if l == "May":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/May_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)
        if l == "June":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/June_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)

        if l =="July":
            # plotting the violin chart
            df = pd.read_csv("https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/July_wqi_2018.csv")
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            #Bar plot
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
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
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


           #Correlation
            if st.checkbox('Correlation plot'):
                plt.figure(figsize=(13, 8))
                sns.heatmap(df.drop(["S.No."], axis=1).corr(), annot=True, cmap='terrain')
                st.pyplot(plt)

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)

        if l == "August":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/August_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)
        if l == "September":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/September_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)
        if l =="October":
            # plotting the violin chart
            df = pd.read_csv("https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/October_wqi_2018.csv")
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            #Bar plot
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
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
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


           #Correlation
            if st.checkbox('Correlation plot'):
                plt.figure(figsize=(13, 8))
                sns.heatmap(df.drop(["S.No."], axis=1).corr(), annot=True, cmap='terrain')
                st.pyplot(plt)

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)

        if l == "November":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/November_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)
        if l == "December":
            # plotting the violin chart
            df = pd.read_csv('https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/December_wqi_2018.csv')
            if st.checkbox('View dataset'):
                st.write(df)

            # Scatter Plot
            fig1 = px.scatter(df, x="District/RO", y="wqi", color="quality", color_discrete_map={
                "Very poor": "red",
                "Excellent": "#17becf",
                "Good": "#72B7B2",
                "Not suitable": '#AF0038',
                "Poor": "#A777F1"},
                              title="<b>WATER QUALITY INDEX</b> ")
            st.plotly_chart(fig1)

            # Bar plot
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
            # st.plotly_chart(fig2)

            # Pie
            # pull is given as a fraction of the pie radius
            fig3 = go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0.1, 0.3, 0.2, 0, 0])])
            # st.plotly_chart(fig3)

            columns = st.columns((2, 1))
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

            # Voilin plot
            if st.checkbox('Statistical Analysis'):
                fig = px.violin(df, x="District/RO", y="wqi")
                st.plotly_chart(fig)
                # st.write(df)



    # This is for Vellore
    if rad == 'Vellore':
        st.subheader("Vellore Water Analysis")

        df = pd.read_csv("https://raw.githubusercontent.com/Somu-cSs/Water-Quality-Analysis-and-Prediction./main/F_Final_vellore_data_wqi.csv")
            # st.write(df)
        if st.checkbox('View dataset'):
                st.write(df)
        fig3 = px.bar(df, x="Date_of_collection", y="wqi", color="quality",
                      color_discrete_map={
                          "Very poor": "red",
                          "Excellent": "#17becf",
                          "Good": "#72B7B2",
                          "Not suitable": '#AF0038',
                          "Poor": "#A777F1"},
                      title="Explicit color mapping")
        st.plotly_chart(fig3)

        fig4 = px.scatter(df, x="Date_of_collection", y="wqi", color="quality", color_discrete_map={
            "Very poor": "red",
            "Excellent": "#17becf",
            "Good": "#72B7B2",
            "Not suitable": '#AF0038',
            "Poor": "#A777F1"},
                          title="<b>WATER QUALITY INDEX!</b> ")
        st.plotly_chart(fig4)

        # Pie
        # pull is given as a fraction of the pie radius
        figp= go.Figure(data=[go.Pie(labels=df.quality, values=df.wqi, pull=[0, 0, 0, 0, 0.4])])
        st.plotly_chart(figp)

        if st.checkbox('Statistical Analysis'):
            fig = px.violin(df, x="Date_of_collection", y="wqi")
            st.plotly_chart(fig)
            k = df.wqi
            st.write('mean:', k.mean())
            # st.write(df)

        fig_s = px.sunburst(df, path=['Date_of_collection', 'wqi'], values='wqi',
                            color='wqi', hover_data=['quality'],
                            color_continuous_scale='bluyl',
                            )
        st.plotly_chart(fig_s, use_container_width=True)

if ra == "Prediction":
    st.title('Water Quality')
    st.image("prediction.jpg")

    # input data
    n_in1 = st.number_input("Enter the pH :")
    n_in2 = st.number_input("Enter the Total Alkaline :")
    n_in3 = st.number_input("Enter the Chloride:")
    n_in4 = st.number_input("Enter the Calcium :")
    n_in5 = st.number_input("Enter the Magnesium :")
    n_in6 = st.number_input("Enter the Sulphate :")
    n_in7 = st.number_input("Enter the Sodium:")
    n_in8 = st.number_input("Enter the Total Dissolved Solids :")
    n_in9 = st.number_input("Enter the Potassium :")
    n_in10 = st.number_input("Enter the F :")

    # st.markdown(n_in1)
    st.markdown(
        f""""

         pH :{n_in1}
         TA :{n_in2}
         Cl :{n_in3}
         Ca :{n_in4}
         Mg :{n_in5}
         Sulphate:{n_in6}
         Na:{n_in7}
         TDS:{n_in8}
         K :{n_in9}
         F : {n_in10}
    """
    )
    loaded_model = pickle.load(open("ridge.pkl", 'rb'))

    # input_d=(8.5,200,250,75,50,250,200,500,12,1)
    input_data = (n_in1, n_in2, n_in3, n_in4, n_in5, n_in6, n_in7, n_in8, n_in9, n_in10)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    # st.write('Water Quality Index',prediction)
    ca = prediction.astype(np.float)
    st.write('WQI:')
    st.markdown(ca)
    if 0< prediction <= 25:
        st.write('Excellent')
    elif 25 < prediction <= 50:
        st.write('Good')
    elif 50 < prediction <= 75:
        st.write('Poor')
    elif 75 < prediction <= 100:
        st.write('Very poor')
    elif prediction > 100:
        st.write('Not suitable')
    # st.wrte(input_data)
