# Agente de Deep Reinforcement Learning (DQN) para Trading Financiero

**Simulación y Aprendizaje por Refuerzo · Unidad 5 · Proyecto Final**
Maestría en Inteligencia Artificial, Universidad de La Sabana · MIA 2026-5
Profesor: Emilio Muñoz Pérez

**Grupo 1**

| Integrante | Correo |
|---|---|
| Gloria Yaned Leguizamon Arboleda | glorialear@unisabana.edu.co |
| Ricardo Sandoval Chaves | ricardosach@unisabana.edu.co |
| Edwin Aviles Cogollo | edwinavco@unisabana.edu.co |
| Orly Alberto Alvao Barajas | orlyalba@unisabana.edu.co |
| Edgar Julian Mendez Ortegon | edgarmenor@unisabana.edu.co |
| Saud Mauricio Gonzalez Rodriguez | saudgoro@unisabana.edu.co |
| Luis Alfonso Fajardo Andrade | luisfaan@unisabana.edu.co |

¡Bienvenidos a nuestro proyecto de aprendizaje por refuerzo! En este repositorio exploramos cómo enseñar a un agente de Inteligencia Artificial a tomar decisiones de inversión en la bolsa de valores (específicamente con acciones de Tesla - TSLA) utilizando **Deep Q-Network (DQN)**.

Hay **dos iteraciones**. El código de esta rama es la segunda. La primera no se borra: queda en el PDF del grupo.

| | Iteración 1 — entrega inicial | Iteración 2 — esta rama |
|---|---|---|
| Informes | [`informes/U5-ProyFinal-AgenteDQN-RL-TradingFinanciero_v1.pdf`](informes/U5-ProyFinal-AgenteDQN-RL-TradingFinanciero_v1.pdf) | [`informes/Informe_AgenteInversor.pdf`](informes/Informe_AgenteInversor.pdf) |
| Capital | $10.000 | $100.000 |
| Reloj | retorno de hoy + trade al close de hoy | retorno de ayer; trade hoy; PnL mañana |
| Optuna | en el repo; no define campeón con val limpia | train/val del train; el test no se usa para elegir |
| DQN / B&H / aleatorio | 2,46 % / 16,27 % / 15,24 % | −1,60 % / +17,25 % / +3,30 % (media 10) |

La iteración 1 es el primer entregable qu eejecutamos. La iteración 2 no “corrige el Excel para que gane”: corrige el método (reloj, valuación, recompensa, Optuna). Un resultado negativo en test, con protocolo mas estricto, sigue siendo un proyecto válido ;) .

---

## Instalación y Uso

Ejecuta **siempre desde la raíz del proyecto** (`trading-rl-project/`). Los scripts leen `configs/` y `data/` con rutas relativas a esa carpeta. Semilla de reproducción: **42** (`configs/tesla_dqn.yaml`).

1. **Clona el repositorio y entra a la carpeta:**
   ```bash
   git clone https://github.com/ricardoschmg/trading-rl-project.git
   cd trading-rl-project
   ```

2. **Activa el entorno Python (recomendado: `Python_MIA`) o crea uno nuevo:**
   ```bash
   # Opción A — entorno del curso (Anaconda):
   conda activate Python_MIA

   # Opción B — entorno virtual local:
   python -m venv env
   # En Windows:
   .\env\Scripts\activate
   # En Mac/Linux:
   source env/bin/activate
   pip install .
   ```

3. **Si usas un entorno nuevo (opción B), instala dependencias** según `pyproject.toml` (Gymnasium, Stable-Baselines3, MLflow, Optuna, PyYAML, etc.):
   ```bash
   pip install .
   ```

4. **Ejecución (en este orden):**
   * Preparar / revisar datos: abrir y ejecutar `notebooks/EDA.ipynb` (Jupyter o VS Code / Cursor). Los CSV ya están en `data/`.
   * Entrenar el agente: `python src/train.py`
   * Evaluar el agente entrenado: `python src/evaluate.py` (requiere `saves/dqn_tesla_model.zip`)
   * Baselines, mismas condiciones de test:
     * `python src/baseline.py` — comprar y mantener
     * `python src/random_agent.py` — agente aleatorio (semilla 42)
   * Sintonizar hiperparámetros en validación (no usa test): `python src/tune.py`
   * Ver panel de métricas: `mlflow ui` (abrir `http://127.0.0.1:5000`)

---

## 1. Formulación del Estado y las Acciones

Para que nuestro agente pueda "ver" el mercado y tomar decisiones sin confundirse por magnitudes muy dispares (precios altos vs. decimales), normalizamos su visión del mundo:

* **El Estado (Observación):** Un vector de 3 dimensiones numéricas:
  1. *Retorno de ayer:* \((P_{t-1}-P_{t-2})/P_{t-2}\). No usa el close de hoy para decidir.
  2. *Balance normalizado:* Efectivo disponible dividido por el capital inicial ($100,000).
  3. *Posición real:* Proporción del portafolio que quedó invertida después del trade (no el objetivo).
* **El Espacio de Acciones:** Se define un espacio discreto (`Discrete(3)`) de asignación de portafolio:
  * `0`: 0% invertido (Quedarse en efectivo / Vender todo).
  * `1`: 50% del capital invertido.
  * `2`: 100% del capital invertido.

## 2. Entorno de Simulación con Costo de Operación

Se crea el entorno personalizado `TradingEnv` usando **Gymnasium**.
* **Reloj de decisión:** en el día \(t\) el agente observa el retorno de ayer, opera al close de hoy y el PnL se marca con el close de mañana. Train (80%) y test (20%) son cronológicos.
* **Fricción de mercado:** comisión del 0.1% (`transaction_fee = 0.001`) sobre el notional de cada rebalanceo. Capital inicial: $100,000.
* **Valuación:** mark-to-market al último close, sin venta forzada. Buy & Hold usa la misma convención.

## 3. Diseño e Iteración de la Función de Recompensa

Se compararon dos \(r_t\) (20.000 steps, solo train):

* `simple`: \((V_{t+1}-V_t)/V_t\) → −1,41 % en train, **99,5 % del tiempo en efectivo**.
* `log`: \(\log(V_{t+1}/V_t)\) → −0,71 % en train, **misma política de cash**.

Se eligió **simple**. Log no se llevó a test: no cambió la política.

## 4. Entrenamiento del Agente (DQN)

* **Algoritmo:** DQN de Stable-Baselines3 (`MlpPolicy`), semilla **42**, 200.000 timesteps, MLflow.
* **Optuna:** 10 trials sobre `learning_rate` y `batch_size`. Cada trial entrena en el 80 % del train y evalúa greedy en el 20 % final (**val**). El test no se usa. Ganador: `lr ≈ 2.75e-5`, `batch_size = 32` (+4,69 % en val). Esos valores quedaron en `configs/tesla_dqn.yaml`.

## 5. Evaluación contra Baselines (Datos Held-out)

Una sola evaluación en `test_tesla.csv`, misma valuación mark-to-market para todos:

| Estrategia | Retorno | Máx. drawdown | Costos |
|---|---|---|---|
| Agente DQN | **−1,60 %** | **−6,16 %** | **$1.405** |
| Comprar y mantener | +17,25 % | −7,18 % | $99,90 |
| Aleatorio (media 10) | +3,30 % | −9,20 % | $2.265 |

Figuras: `saves/equity_curve.png`, `saves/dqn_positions.png`.

## 6. Discusión y Reflexión

* **No-estacionariedad:** train bajista (−27 %) y test alcista (+16 %). El agente aprendió a proteger capital y no se subió al rally. Por eso pierde contra Buy & Hold.
* **Reward hacking suave:** en train, quedarse en cash maximiza \(r\). Simple y log produjeron esa misma conducta.
* **Protocolo:** no se restauró un checkpoint anterior que en test daba +2,78 %. Elegirlo ahora usaría el held-out para seleccionar el campeón.
* **Informes (carpeta `informes/`):**
  * Iteración 1: `U5-ProyFinal-AgenteDQN-RL-TradingFinanciero_v1.pdf`
  * Iteración 2: `Informe_AgenteInversor.pdf`

