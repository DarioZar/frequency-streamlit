import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import peakutils
from zipfile import ZipFile

def fourier_transform(df, dt):
    df_fft = df.loc[:, "A_x":"A_mean"].apply(np.fft.rfft, axis=0).apply(np.abs, axis=0).apply(np.square, axis=0)
    df_fft.loc[:,"A_x":"A_mean"] /= df_fft.loc[:,"A_x":"A_mean"].apply(np.max, axis=0)
    df_fft.columns = "FFT_" + df_fft.columns
    df_fft.insert(0, "freq", np.fft.rfftfreq(df["t"].size, dt))
    return df_fft

st.title("Signal analyzer")
st.sidebar.title("Options")
datafile = st.sidebar.file_uploader("File", type=["xls", "zip"], key=None, help="Carica il file di dati",)

if datafile is not None:
    try:
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

        df = readfile(datafile, header=0, names=["t", "A_x", "A_y", "A_z", "A_mean"])
        df["A_mean"] = df.loc[:, "A_x":"A_z"].mean(axis=1)
        dt = df["t"][0]
        tmin, tmax = float(df["t"].iat[0]), float(df["t"].iat[-1])

        option = st.sidebar.selectbox('What do you want to plot?',
                              ('A_x', 'A_y', 'A_z', 'A_mean'))
        range =  st.sidebar.slider("T range", min_value=tmin, max_value=tmax, value=(tmin,tmax))

        df = df.loc[df["t"].between(range[0], range[1])].reset_index(drop=True)
        df_fft = fourier_transform(df, dt)

        st.subheader("Signal")
        st.line_chart(df[["t",option]].rename(columns={'t':'index'}).set_index('index'))
        st.subheader("Signal PSD")
        st.line_chart(df_fft[["freq","FFT_"+option]].rename(columns={'freq':'index'}).set_index('index')[1:])
        #peak = df_fft["freq"][np.argmax(df_fft["FFT_"+option][1:])]
        #st.write(f"La frequenza di picco corrisponde a {peak:.3f} Hz")
        indexes = peakutils.indexes(df_fft["FFT_"+option][1:], thres=0.5)
        for i in indexes:
            st.write(f"Picco a {df_fft['freq'][i]:.3f} Hz)")

    except TypeError:
        st.write("Invalid file")