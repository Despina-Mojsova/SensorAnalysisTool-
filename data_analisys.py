import tkinter as tk
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog, messagebox
from sklearn.ensemble import IsolationForest
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

class AnomalyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Detection App")
        self.root.geometry("450x360")

        # Section: User Interface Buttons
        tk.Label(root, text="Select an analysis model:").pack(pady=10)

        tk.Button(root, text="Isolation Forest", command=lambda: self.start_analysis("isolation"), width=30).pack(pady=5)
        tk.Button(root, text="Autoencoder", command=lambda: self.start_analysis("autoencoder"), width=30).pack(pady=5)
        tk.Button(root, text="Frequency Analysis", command=self.frequency_analysis, width=30).pack(pady=5)
        tk.Button(root, text="Filter Noise (MATLAB)", command=self.filter_noise_matlab, width=30).pack(pady=5)
        tk.Button(root, text="Generate Report", command=self.generate_report, width=30).pack(pady=5)
        tk.Button(root, text="Exit", command=root.quit, width=30).pack(pady=10)

    # Section: Start analysis based on selected model
    def start_analysis(self, model_type):
        file_path = filedialog.askopenfilename(title="Select sensor data", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        data = pd.read_csv(file_path)
        if model_type == "isolation":
            self.isolation_forest_analysis(data)
        elif model_type == "autoencoder":
            self.autoencoder_analysis(data)

    # Section: MATLAB noise filtering integration
    def filter_noise_matlab(self):
        file_path = filedialog.askopenfilename(title="Select sensor data", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        filtered_output = "filtered_output.csv"

        try:
            # Call MATLAB script for noise filtering
            subprocess.run([
                "matlab", "-batch",
                f"filter_signal('{file_path.replace('\\', '/')}', '{filtered_output}')"
            ], check=True)

            messagebox.showinfo("Success", "Noise filtered successfully using MATLAB.")
            filtered_data = pd.read_csv(filtered_output)

            # Plot original and filtered signal
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_data["Time"], filtered_data["Sensor Value"], label="Original", alpha=0.5)
            plt.plot(filtered_data["Time"], filtered_data["Filtered Sensor Value"], label="Filtered", linewidth=2)
            plt.legend()
            plt.title("Noise Filtering (MATLAB)")
            plt.xlabel("Time (s)")
            plt.ylabel("Sensor Value")
            plt.grid(True)
            plt.show()

        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"MATLAB failed: {e}")

    # Section: Isolation Forest anomaly detection
    def isolation_forest_analysis(self, data):
        model = IsolationForest(contamination=0.02, random_state=42)
        data['Anomaly'] = model.fit_predict(data[['Sensor Value']])
        data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})
        data.to_csv("anomaly_results.csv", index=False)
        self.plot_results(data, "Isolation Forest")

    # Section: Autoencoder anomaly detection
    def autoencoder_analysis(self, data):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(data[['Sensor Value']])

        autoencoder = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(X_train.shape[1], activation='sigmoid')
        ])

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)

        X_pred = autoencoder.predict(X_train)
        mse = np.mean(np.power(X_train - X_pred, 2), axis=1)
        threshold = np.percentile(mse, 95)
        data['Autoencoder_Anomaly'] = mse > threshold
        data.to_csv("anomaly_results.csv", index=False)
        self.plot_results(data, "Autoencoder")

    # Section: FFT frequency spectrum analysis
    def frequency_analysis(self):
        file_path = filedialog.askopenfilename(title="Select sensor data", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        data = pd.read_csv(file_path)
        signal = data['Sensor Value'].values
        N = len(signal)
        T = 1.0 / 800.0  # Sampling period assumed 800 Hz
        x = np.linspace(0.0, N * T, N, endpoint=False)
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, T)[:N // 2]

        plt.figure(figsize=(10, 5))
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    # Section: Visualize anomalies on plot
    def plot_results(self, data, model_name):
        if 'Anomaly' in data:
            anomalies = data[data['Anomaly'] == 1]
        elif 'Autoencoder_Anomaly' in data:
            anomalies = data[data['Autoencoder_Anomaly'] == True]
        else:
            anomalies = pd.DataFrame()

        plt.figure(figsize=(10, 5))
        plt.plot(data['Time'], data['Sensor Value'], label="Sensor Data")
        plt.scatter(anomalies['Time'], anomalies['Sensor Value'], color='red', label="Anomalies", zorder=3)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.title(f'Anomaly Detection with {model_name}')
        plot_filename = "anomaly_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        return plot_filename

    # Section: Calculate and display anomaly percentage
    def show_anomaly_percentage(self, data, model_name):
        if 'Anomaly' in data:
            anomaly_count = data['Anomaly'].sum()
        elif 'Autoencoder_Anomaly' in data:
            anomaly_count = data['Autoencoder_Anomaly'].sum()
        else:
            anomaly_count = 0

        total_count = len(data)
        anomaly_percentage = (anomaly_count / total_count) * 100

        messagebox.showinfo("Result", f"Anomaly percentage with {model_name}: {anomaly_percentage:.2f}%")
        return anomaly_percentage

    # Section: Generate PDF report with results
    def generate_report(self):
        report_filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not report_filename:
            return

        data = pd.read_csv("anomaly_results.csv")
        anomaly_percentage = self.show_anomaly_percentage(data, "Anomaly Detection")
        plot_filename = self.plot_results(data, "Anomaly Detection")

        c = canvas.Canvas(report_filename, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica", 12)
        c.drawString(100, height - 50, f"Anomaly Detection Report")
        c.drawString(100, height - 80, f"Anomaly percentage: {anomaly_percentage:.2f}%")
        c.drawString(100, height - 110, "Anomaly plot:")
        c.drawImage(plot_filename, 100, height - 400, width=400, height=200)
        c.showPage()
        c.save()

        messagebox.showinfo("Success", f"Report successfully generated: {report_filename}")

# App entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    root.mainloop()
