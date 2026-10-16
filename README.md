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

Este proyecto fue desarrollado con fines académicos para la clase de Reinforcement Learning (RL). A continuación, documentamos paso a paso proceso, experimentos y lecciones aprendidas.

---

## Instalación y Uso

El proyecto está modularizado y utiliza un entorno virtual estándar. Para ejecutarlo en tu máquina, sigue estos pasos:

1. **Clona el repositorio y entra a la carpeta:**
   ```bash
   git clone https://github.com/tu-usuario/rl_trading_project.git
   cd rl_trading_project
   ```

2. **Crea y activa un entorno virtual:**
   ```bash
   python -m venv env
   # En Windows:
   .\env\Scripts\activate
   # En Mac/Linux:
   source env/bin/activate
   ```

3. **Instala las dependencias:**
   De acuerdo al archivo `pyproject.toml`, se instalan dependencias (incluye Gymnasium, Stable-Baselines3, MLflow, Optuna, etc.):
   ```bash
   pip install .
   ```

4. **Ejecución de scripts principales:**
   * Descargar datos y preparar datasets: `python src/EDA.ipynb` (o ejecutar el notebook).
   * Entrenar el agente: `python src/train.py`
   * Evaluar el agente entrenado: `python src/evaluate.py`
   * Sintonizar hiperparámetros: `python src/tune.py`
   * Ver panel de métricas: `mlflow ui` (y abrir `http://127.0.0.1:5000`)

---

## 1. Formulación del Estado y las Acciones

Para que nuestro agente pueda "ver" el mercado y tomar decisiones sin confundirse por magnitudes muy dispares (precios altos vs. decimales), normalizamos su visión del mundo:

* **El Estado (Observación):** Un vector de 3 dimensiones numéricas:
  1. *Retorno porcentual diario del precio:* ¿Cuánto subió o bajó el precio hoy respecto a ayer?
  2. *Balance normalizado:* Efectivo disponible dividido por el capital inicial ($10,000).
  3. *Posición actual:* Proporción del portafolio actualmente invertida.
* **El Espacio de Acciones:** Se define un espacio discreto (`Discrete(3)`) de asignación de portafolio:
  * `0`: 0% invertido (Quedarse en efectivo / Vender todo).
  * `1`: 50% del capital invertido.
  * `2`: 100% del capital invertido.

## 2. Entorno de Simulación con Costo de Operación

Se Crea entorno personalizado `TradingEnv` usando la librería **Gymnasium**. 
* **Dinámica temporal:** El entorno itera sobre datos históricos cronológicos de Tesla divididos en Train (80%) y Test (20%) para evitar mirar al futuro (*look-ahead bias*).
* **Fricción de Mercado (Comisiones):** se implementamo un costo por operar del 0.1% (`transaction_fee = 0.001`). Cada vez que el agente cambia su porcentaje de inversión (acción), el entorno calcula el valor del trade y cobra automáticamente la comisión.

## 3. Diseño e Iteración de la Función de Recompensa

* **La Recompensa:** Se definió como el *cambio porcentual del patrimonio neto (Net Worth)* diario. Si el agente toma decisiones que hacen crecer el portafolio (descontando comisiones), recibe un premio; si pierde valor, recibe un castigo.

## 4. Entrenamiento del Agente (DQN)

* **Algoritmo y Herramientas:** Utilizamos el algoritmo **DQN** de la librería *Stable-Baselines3*. Se Configuran hiperparámetros desde el archivo `tesla_dqn.yaml` . Se usa **MLflow** para registrar cada experimento.
* **Optimización:** Integramos **Optuna** para buscar la mejor combinación de tasa de aprendizaje (`learning_rate`) y tamaño de lote (`batch_size`), dándonos un enfoque científico en lugar de probar al azar.

## 5. Evaluación contra Baselines (Datos Held-out)

Evaluamos a nuestro modelo campeón sobre el 20% de los datos que jamás había visto (`test_tesla.csv`). Para saber si el agente realmente es inteligente, lo comparamos con dos estrategias base:

1. **Baseline - Comprar y Mantener (*Buy & Hold*):** Ganó un **16.27%**. El periodo de prueba fue fuertemente alcista.
2. **Baseline - Agente Aleatorio:** Ganó un **15.24%** (por pura inercia del mercado alcista, tomando decisiones al azar).
3. **Nuestro Agente DQN:** Logró un **2.46%** asumiendo las comisiones, y un **4.50%** en un experimento sin cobro de comision por operacion realizada. 

## 6. Discusión y Reflexión

* **El peso de las comisiones:** Nuestro agente ganó menos que un *Buy & Hold* pasivo. Se visualiza con estor que en mercados explosivamente alcistas, operar frecuentemente es castigado duramente por las comisiones. 
* **Aversión al riesgo:** La política que aprendió nuestro modelo DQN prioriza proteger el capital en escenarios de volatilidad, operando de forma conservadora.
* **Estudios de ablación:** Al entrenar al agente con la comisión en `0.0`, vimos un aumento inmediato en el rendimiento, demostrando que la fricción del mercado es el principal freno de los agentes de trading automatizado.

