import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import time
import datetime
from scipy.stats import binned_statistic
from scipy.stats import gaussian_kde

# Configurer Streamlit


# Fonction pour vérifier les identifiants
def check_password():
    def password_entered():
        if (st.session_state["username"] == "oss" and 
            st.session_state["password"] == "2650"):
            st.session_state["auth"] = True
        else:
            st.session_state["auth"] = False
            st.error("Nom d'utilisateur ou mot de passe incorrect")

    if "auth" not in st.session_state:
        st.session_state["auth"] = False

    if not st.session_state["auth"]:
        st.text_input("Nom d'utilisateur", key="username")
        st.text_input("Mot de passe", type="password", key="password")
        st.button("Se connecter", on_click=password_entered)
        return False
    return True

if check_password(): 
    st.set_page_config(layout="wide")
    st.title("Application de Prédiction des Prix")

    # Liste des symboles disponibles
    symbols = ['GC=F', '^DJI', 'NQ=F', 'BTC-USD', 'EURUSD=X', 'JPY=X', '^GSPC ']
    symbol = st.sidebar.selectbox("Choisir un symbole", symbols)

    # Calculer la date actuelle, la date de demain et la date d'hier
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    yesterday = today - datetime.timedelta(days=1)

    # Configurer les dates de début et de fin avec les valeurs par défaut
    start_date = st.sidebar.date_input("Date de début", value=yesterday)
    end_date = st.sidebar.date_input("Date de fin", value=tomorrow)

    # Paramètres de l'application

    interval = st.sidebar.selectbox("Intervalle", ['1m','5m', '15m', '30m', '1h','4h','1d'])

    # Ajout d'options pour ajuster les hyperparamètres du modèle LSTM
    st.sidebar.write("Paramètres Avancés")
    lstm_units = st.sidebar.slider("Units for LSTM layers", 10, 100, 50)
    lstm_epochs = st.sidebar.slider("Epochs for LSTM", 1, 20, 5)
    batch_size = st.sidebar.slider("Batch Size for LSTM", 1, 32, 1)

    # Fonction pour calculer le MACD
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    # Fonction pour calculer le Volume Profile
    #def calculate_volume_profile(data, num_bins=100):
    #    price_bins = np.linspace(data['Low'].min(), data['High'].max(), num_bins)
    #    volume_profile = np.zeros(num_bins - 1)
    #    for i in range(num_bins - 1):
    #        bin_mask = (data['Close'] >= price_bins[i]) & (data['Close'] < price_bins[i+1])
    #        volume_profile[i] = data.loc[bin_mask, 'Volume'].sum()
    #    return price_bins, volume_profile

    # Calcul du Volume Profile
    def calculate_volume_profile(data, num_bins=150):
        price_data = data['Close']
        volume_data = data['Volume']
        bin_edges = np.linspace(price_data.min(), price_data.max(), num_bins + 1)
        volume_profile, bin_edges, _ = binned_statistic(price_data, volume_data, statistic='sum', bins=bin_edges)
        return volume_profile, bin_edges

    # Fonction pour afficher les KPI
    #def display_kpis(data):
    #    close_price = data['Close'].iloc[-1]
    #    price_difference = data['Close'].iloc[-1] - data['Close'].iloc[0]
    #    high_price = data['High'].max()
    #    low_price = data['Low'].min()
    #    volume_actual = data['Volume'].iloc[-1]

    #    col1, col2, col3, col4, col5 = st.columns(5)
    #    with col1:
    #        st.metric(label="Prix de Clôture", value=f"${close_price:.2f}")
    #    with col2:
    #        st.metric(label="Différence de Prix", value=f"${price_difference:.2f}")
    #    with col3:
    #        st.metric(label="Prix le Plus Élevé", value=f"${high_price:.2f}")
    #    with col4:
    #        st.metric(label="Prix le Plus Bas", value=f"${low_price:.2f}")
    #    with col5:
    #        st.metric(label="Volume" ,value=f"{volume_actual:.2f}")

    # Fonction pour afficher les KPI
   

    # Fonction pour afficher les KPI
    def display_kpis(data):
        close_price = data['Close'].iloc[-1]
        price_difference = data['Close'].iloc[-1] - data['Close'].iloc[0]
        high_price = data['High'].max()
        low_price = data['Low'].min()
        data_size = data.shape[0]

        # Styles CSS pour centrer et ajouter une bordure réduite
        kpi_style = """
        <style>
        .kpi {
            text-align: center;
            border: 1px solid #eab676;
            border-radius: 5px;
            padding: 10px;
            margin: 10px;
            flex: 1; /* Assure que les KPI occupent une largeur égale */
        }
        .kpi-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap; /* Permet aux KPI de passer à la ligne si l'espace est insuffisant */
        }
        </style>
        """

        st.markdown(kpi_style, unsafe_allow_html=True)

        # Utilisation de st.columns pour aligner les KPI verticalement
        col1, col2, col3, col4, col5 = st.columns(5)  # Crée deux colonnes, la deuxième étant plus large

        with col1:
            st.markdown('<div class="kpi"><strong>Prix de Clôture</strong><br>${:.2f}</div>'.format(close_price), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="kpi"><strong>Différence de Prix</strong><br>${:.2f}</div>'.format(price_difference), unsafe_allow_html=True)
        with col3:        
            st.markdown('<div class="kpi"><strong>Prix le Plus Élevé</strong><br>${:.2f}</div>'.format(high_price), unsafe_allow_html=True)
        with col4:    
            st.markdown('<div class="kpi"><strong>Prix le Plus Bas</strong><br>${:.2f}</div>'.format(low_price), unsafe_allow_html=True)
        with col5:    
            st.markdown('<div class="kpi"><strong>Lignes</strong><br>{:.2f}</div>'.format(data_size), unsafe_allow_html=True)

    # Bouton pour lancer la prédiction
    if st.sidebar.button("Predict"):
        # Affichage d'une animation de chargement
        with st.spinner("Prédiction en cours..."):
            # Télécharger les données
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            data.index = data.index.tz_localize(None)
            # Vérification des données manquantes
            if data.isnull().values.any():
                st.error("Les données téléchargées contiennent des valeurs manquantes. Veuillez vérifier les paramètres ou choisir un autre symbole.")
            else:
                # Calcul des indicateurs techniques
                def calculate_RSI(data, window=14):
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                    RS = gain / loss
                    RSI = 100 - (100 / (1 + RS))
                    return RSI

                def calculate_z_score(data):
                    mean = data['Close'].rolling(window=20).mean()
                    std = data['Close'].rolling(window=20).std()
                    z_score = (data['Close'] - mean) / std
                    return z_score

                data['ZScore'] = calculate_z_score(data)
                data['RSI'] = calculate_RSI(data)
                data['MACD_Line'], data['Signal_Line'], data['MACD_Histogram'] = calculate_macd(data)

                # Afficher les KPI
                display_kpis(data)

                # Afficher les dernières données
                st.write("Aperçu des dernières données:")
                #st.write(data.tail(5))
                st.dataframe(data.tail(5), use_container_width=True)

                # Calculer les rendements
                data['Return'] = data['Close'].pct_change()
                data = data.dropna()

                # Ajuster un modèle SARIMA aux rendements
                sarima_model = SARIMAX(data['Return'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 5))
                sarima_fit = sarima_model.fit(disp=False)

                # Prédire les rendements pour les 30 prochaines minutes
                forecast_steps_30 = 30
                forecast_30 = sarima_fit.get_forecast(steps=forecast_steps_30)
                forecast_Index_30 = [data.index[-1] + pd.Timedelta(minutes=i) for i in range(1, forecast_steps_30 + 1)]
                forecast_values_30 = forecast_30.predicted_mean

                # Prédire les prix en utilisant les rendements prédits par SARIMA
                predicted_prices_30 = []
                last_price_30 = data['Close'].iloc[-1]
                for return_30 in forecast_values_30:
                    predicted_price_30 = last_price_30 * (1 + return_30)
                    predicted_prices_30.append(predicted_price_30)
                    last_price_30 = predicted_price_30

                # Simuler les prix en utilisant la méthode de Monte Carlo
                n_simulations = 10000
                simulated_trajectories = []
                mean_return = data['Return'].mean()
                std_return = data['Return'].std()
                for _ in range(n_simulations):
                    simulated_returns = np.random.normal(mean_return, std_return, forecast_steps_30)
                    simulated_prices = [data['Close'].iloc[-1]]
                    for return_ in simulated_returns:
                        simulated_price = simulated_prices[-1] * (1 + return_)
                        simulated_prices.append(simulated_price)
                    simulated_trajectories.append(simulated_prices)

                # Ajuster la longueur des trajectoires simulées
                simulated_trajectories = [trajectory[:forecast_steps_30] for trajectory in simulated_trajectories]
                mse_values = [mean_squared_error(predicted_prices_30, trajectory) for trajectory in simulated_trajectories]

                # Trouver la trajectoire simulée avec l'erreur quadratique moyenne la plus faible
                min_mse_index = np.argmin(mse_values)
                closest_trajectory = simulated_trajectories[min_mse_index]

                # Préparation des données pour LSTM
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data[['Close', 'RSI', 'ZScore', 'MACD_Line', 'Signal_Line']].dropna())

                # Créer les séquences avec features (Close, RSI, ZScore, MACD_Line, Signal_Line)
                def create_sequences_with_features(data, seq_length):
                    x = []
                    y = []
                    for i in range(seq_length, len(data)):
                        x.append(data[i-seq_length:i, :])
                        y.append(data[i, 0])
                    x, y = np.array(x), np.array(y)
                    # Assurez-vous que x a trois dimensions
                    if x.ndim == 2:
                        x = np.expand_dims(x, axis=1)
                    return x, y

                seq_length = 60
                x_train, y_train = create_sequences_with_features(scaled_data, seq_length)

                # Vérifiez les dimensions avant de faire le reshape
                if len(x_train.shape) == 3:
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
                else:
                    st.error("Problème avec les dimensions de x_train. Vérifiez les données d'entrée.")

                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

                # Modèle LSTM
                model = Sequential()
                model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(LSTM(units=lstm_units, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))

                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=batch_size, epochs=lstm_epochs)

                last_60_days = scaled_data[-60:]
                lstm_input = last_60_days.reshape(1, last_60_days.shape[0], last_60_days.shape[1])
                lstm_predicted_prices = []

                for _ in range(forecast_steps_30):
                    lstm_predicted_price = model.predict(lstm_input)
                    lstm_predicted_prices.append(lstm_predicted_price[0][0])
                    lstm_predicted_price_reshaped = np.full((1, 1, lstm_input.shape[2]), lstm_predicted_price[0][0])
                    lstm_input = np.append(lstm_input[:, 1:, :], lstm_predicted_price_reshaped, axis=1)

                lstm_predicted_prices = np.array(lstm_predicted_prices).reshape(-1, 1)
                lstm_predicted_prices_full = np.zeros((lstm_predicted_prices.shape[0], scaled_data.shape[1]))
                lstm_predicted_prices_full[:, 0] = lstm_predicted_prices[:, 0]

                lstm_predicted_prices = scaler.inverse_transform(lstm_predicted_prices_full)[:, 0]

                # Calculer le Volume Profile
                #price_bins, volume_profile = calculate_volume_profile(data)


                # Calculer le Volume Profile
                volume_profile, bin_edges = calculate_volume_profile(data)


                # Visualisation interactive avec Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Prices',line=dict(color='white') , showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_Index_30, y=predicted_prices_30, mode='lines',  name='SARIMA',line=dict(color='red'), showlegend=False ))
                fig.add_trace(go.Scatter(x=forecast_Index_30, y=closest_trajectory, mode='lines', name='Monte Carlo',line=dict(color='green'),showlegend=False ))
                fig.add_trace(go.Scatter(x=forecast_Index_30, y=lstm_predicted_prices, mode='lines', name='LSTM',line=dict(color='#eab676'), showlegend=False ))
                fig.update_layout(title=f'Price Prediction for {symbol}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)
            

                # Visualisation du MACD
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Line'], mode='lines', name='MACD Line', line=dict(color='white'),showlegend=False))
                fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#eab676'), showlegend=False))
                fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='MACD Histogram', marker_color='grey', opacity=0.7,showlegend=False))
                fig_macd.update_layout(title=f'MACD for {symbol}', xaxis_title='Date', yaxis_title='MACD', xaxis=dict(tickformat="%d-%m-%Y"))
                st.plotly_chart(fig_macd)
            

                # Visualisation du RSI
                
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='white'),showlegend=False))
                fig_rsi.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines', name='Seuil 30', line=dict(color='#eab676', dash='dash'),showlegend=False))
                fig_rsi.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines', name='Seuil 70', line=dict(color='#eab676', dash='dash'),showlegend=False))
                fig_rsi.update_layout(title=f'RSI for {symbol}', xaxis_title='Date', yaxis_title='RSI', xaxis=dict(tickformat="%d-%m-%Y"))
                st.plotly_chart(fig_rsi)

                # Préparation des traces pour le graphique des prix et le Volume Profile
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=volume_profile,  # Densité normalisée
                    y=bin_edges[:-1],  # Limites inférieures des bacs
                    orientation='h',
                    marker=dict(color='white', opacity=0.9),
                    name='Volume Profile'
                ))
                volume_fig.update_layout(
                    title='Volume Profile',
                    xaxis_title='Volume',
                    yaxis_title='Price',
                    yaxis=dict(tickformat=".2f"),
                    height=600
                )

                # Affichage du graphique avec Streamlit
                st.plotly_chart(volume_fig, use_container_width=True)

                # Option de téléchargement des résultats
                st.write("Télécharger les prédictions:")
                csv = data.to_csv(index=True)
                st.download_button("Télécharger les données", data=csv, file_name=f'{symbol}_predictions.csv', mime='text/csv')

