import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
import peakutils
#from scipy.signal import find_peaks
from zipfile import ZipFile

#TODO: dark theme for bokeh
#def bokeh_figure_dark():
#    pass

def fourier_transform(df, dt):
    # Calc PSD of acceleration signal for every axis
    df_fft = df.loc[:, "A_x":"A_mean"].apply(np.fft.rfft, axis=0).apply(np.abs, axis=0).apply(np.square, axis=0)
    # Rename columns
    df_fft.columns = "FFT_" + df_fft.columns
    # Calc frequencies
    df_fft.insert(0, "freq", np.fft.rfftfreq(df["t"].size, dt))
    # Drop first row (peak at zero)
    df_fft = df_fft.iloc[1: , :]
    # Normalize to 1
    df_fft.loc[:,"FFT_A_x":"FFT_A_mean"] /= df_fft.loc[:,"FFT_A_x":"FFT_A_mean"].apply(np.max, axis=0)
    return df_fft

st.title("Signal analyzer")
st.sidebar.title("Options")
# File uploader for phyphox file formats
datafile = st.sidebar.file_uploader("File", type=["xls", "zip"], key=None, help="Carica il file di dati",)

if datafile is not None:
    try:
        # Check if file is in phyphox format (excel or zip with csv)
        # If not throw exception
        if datafile.type == "application/vnd.ms-excel":
            readfile = pd.read_excel
        elif datafile.type == "application/zip":
            zipped = ZipFile(datafile)
            if 'Raw Data.csv' in zipped.namelist():
                datafile = zipped.open( 'Raw Data.csv')
                readfile = pd.read_csv
            else:
                raise TypeError
        else:
            raise TypeError

        # Read acceleration signal, discard |A| and calc A_mean
        df = readfile(datafile, header=0, names=["t", "A_x", "A_y", "A_z", "A_mean"])
        df["A_mean"] = df.loc[:, "A_x":"A_z"].mean(axis=1)
        # Calc dt, t_min and t_max
        dt = df["t"][1]-df["t"][0]
        tmin, tmax = float(df["t"].iat[0]), float(df["t"].iat[-1])

        # Create options widgets
        option = st.sidebar.selectbox('What do you want to plot?',
                              ('A_x', 'A_y', 'A_z', 'A_mean'))
        range =  st.sidebar.slider("T range", min_value=tmin, max_value=tmax, value=(tmin,tmax))

        # Resize data from range slider, calc PSD
        df = df.loc[df["t"].between(range[0], range[1])].reset_index(drop=True)
        df_fft = fourier_transform(df, dt)

        # Plot data
        st.subheader("Signal")
        ## using integrated functions x must be index of DataFrame
        #st.line_chart(df[["t",option]].rename(columns={'t':'index'}).set_index('index'))
        p1 = figure(width=450, height=350, x_axis_label='t (s)', y_axis_label=option+" (m/s^2)")
        p1.line(df["t"],df[option])
        st.bokeh_chart(p1, use_container_width=True)
        st.subheader("Signal PSD")
        #st.line_chart(df_fft[["freq","FFT_"+option]].rename(columns={'freq':'index'}).set_index('index')[1:])
        p2 = figure(width=450, height=350, x_axis_label='f (Hz)', y_axis_label=option+" PSD")
        p2.line(df_fft["freq"],df_fft["FFT_"+option])
        st.bokeh_chart(p2, use_container_width=True)
            #peak = df_fft["freq"].iat[np.argmax(df_fft["FFT_"+option])]
            #st.write(f"La frequenza di picco corrisponde a {peak:.3f} Hz")
        # Calc peaks using peakutils
        print(df_fft)
        indexes = peakutils.indexes(df_fft["FFT_"+option])
        #indexes, _ = find_peaks(df_fft["FFT_"+option], height=0.3)
        for i in indexes:
            st.write(f"Peak: {df_fft['freq'].iat[i]:.3f} Hz")

    except TypeError:
        st.write("Invalid file. File must be in Phyphox format.")