import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def tanh(x):
    return np.tanh(x)


def deriv_tanh(x):
    return 1 - x**2


# Multiplikation
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
inp = np.c_[X.ravel(), Y.ravel()]
target = (X * Y).reshape(-1, 1)

# Die Architektur des neuronalen Netzes
inp_size = 2  # Eingabeneuronen
hid_size = 4  # Hidden-Neuronen
out_size = 1  # Ausgabeneuron

# Gewichte zufällig initialisieren (Mittelwert = 0)
w0 = np.random.uniform(-0.5, 0.5, (inp_size, hid_size))
w1 = np.random.uniform(-0.5, 0.5, (hid_size, out_size))

b0 = np.random.uniform(-0.1, 0.1, (1, hid_size))
b1 = np.random.uniform(-0.1, 0.1, (1, out_size))

m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
m_w1, v_w1 = np.zeros_like(w1), np.zeros_like(w1)
m_b0, v_b0 = np.zeros_like(b0), np.zeros_like(b0)
m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 1e-2

# Netzwerk trainieren
for i in range(1, 20001):  # Start from 1 for Adam bias correction
    if i % 1000 == 0:
        print(f"Iteration: {i}")
    # Vorwärtsaktivierung
    L0 = inp
    L1 = tanh(np.matmul(L0, w0) + b0)
    L2 = np.matmul(L1, w1) + b1

    # Fehler berechnen
    L2_error = target - L2

    # Backpropagation
    L2_delta = L2_error
    L1_error = np.matmul(L2_delta, w1.T)
    L1_delta = L1_error * deriv_tanh(L1)

    N = inp.shape[0]
    grad_w1 = np.matmul(L1.T, L2_delta) / N
    grad_w0 = np.matmul(L0.T, L1_delta) / N
    grad_b1 = np.sum(L2_delta, axis=0, keepdims=True) / N
    grad_b0 = np.sum(L1_delta, axis=0, keepdims=True) / N

    m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1
    m_w0 = beta1 * m_w0 + (1 - beta1) * grad_w0
    m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1
    m_b0 = beta1 * m_b0 + (1 - beta1) * grad_b0

    v_w1 = beta2 * v_w1 + (1 - beta2) * (grad_w1**2)
    v_w0 = beta2 * v_w0 + (1 - beta2) * (grad_w0**2)
    v_b1 = beta2 * v_b1 + (1 - beta2) * (grad_b1**2)
    v_b0 = beta2 * v_b0 + (1 - beta2) * (grad_b0**2)

    m_w1_hat = m_w1 / (1 - beta1**i)
    m_w0_hat = m_w0 / (1 - beta1**i)
    m_b1_hat = m_b1 / (1 - beta1**i)
    m_b0_hat = m_b0 / (1 - beta1**i)

    v_w1_hat = v_w1 / (1 - beta2**i)
    v_w0_hat = v_w0 / (1 - beta2**i)
    v_b1_hat = v_b1 / (1 - beta2**i)
    v_b0_hat = v_b0 / (1 - beta2**i)

    # Gewichte / Bias aktualisieren
    w1 += learning_rate * m_w1_hat / (np.sqrt(v_w1_hat) + epsilon)
    w0 += learning_rate * m_w0_hat / (np.sqrt(v_w0_hat) + epsilon)
    b1 += learning_rate * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)
    b0 += learning_rate * m_b0_hat / (np.sqrt(v_b0_hat) + epsilon)

# Netzwerk testen
L0 = inp
L1 = tanh(np.matmul(inp, w0) + b0)
L2 = np.matmul(L1, w1) + b1

predicted = L2.reshape(X.shape)

pio.renderers.default = "browser"
fig = go.Figure(data=[go.Surface(x=x, y=y, z=predicted, colorscale="greens")])
fig.show()
