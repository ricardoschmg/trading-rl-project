import pandas as pd

def calculate_buy_and_hold():
    # Cargar los datos de prueba
    df = pd.read_csv("data/test_tesla.csv")
    initial_balance = 10000.0
    fee = 0.001

    # Día 1: Comprar con todo el capital disponible (restando la comisión)
    first_price = float(df.iloc[0]['close'])
    investable_capital = initial_balance / (1 + fee)
    shares_held = investable_capital / first_price

    # Último día: Vender todas las acciones (restando la comisión)
    last_price = float(df.iloc[-1]['close'])
    gross_proceeds = shares_held * last_price
    final_balance = gross_proceeds * (1 - fee)

    # Calcular retorno porcentual
    return_pct = ((final_balance - initial_balance) / initial_balance) * 100

    print("=== BASELINE: COMPRAR Y MANTENER (BUY & HOLD) ===")
    print(f"Precio de compra inicial: ${first_price:.2f}")
    print(f"Precio de venta final:    ${last_price:.2f}")
    print(f"Balance Inicial:          ${initial_balance:,.2f}")
    print(f"Balance Final:            ${final_balance:,.2f}")
    print(f"Retorno Total:            {return_pct:.2f}%")

if __name__ == "__main__":
    calculate_buy_and_hold()