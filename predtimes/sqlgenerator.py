"""
This script generates a SQL file with synthetic time series data for sales.

The generated data can be used for testing and development of the PredTimes
application. The script creates a large number of sales records with
randomized but plausible data, including seasonality effects.
"""
import random
from datetime import datetime, timedelta

# Configuration
NUM_DIAS = 365 * 8  # Años de datos
FECHA_INICIO = datetime(2000, 5, 1)
ARCHIVO_SALIDA = "ventas_entrenamiento.sql"

# Products (Name, Price, Sales probability weight)
# A higher weight means the product is sold more frequently (e.g., cables vs. laptops)
productos = [
    ("Laptop Modelo X", 1200.50, 5),
    ("Mouse Inalámbrico", 25.00, 30),
    ("Teclado Mecánico", 89.99, 15),
    ("Monitor 27\"", 299.90, 10),
    ("Cable HDMI", 12.50, 40),
    ("Silla de Oficina", 150.00, 8),
    ("Disco Duro SSD 1TB", 99.99, 20),
    ("Memoria RAM 16GB", 65.00, 25),
    ("Webcam HD", 45.75, 15),
    ("Auriculares Bluetooth", 79.99, 20),
    ("Soporte Monitor Doble", 55.00, 10),
    ("Alfombrilla XL", 19.99, 35),
    ("Caja Externa SSD", 22.00, 15),
    ("Maletín Laptop", 35.50, 12)
]


def generar_sql():
    """
    Generates the SQL script with synthetic sales data.

    Returns:
        A string containing the complete SQL script.
    """
    sql_lines = []

    # SQL Header
    sql_lines.append("-- --------------------------------------------------------")
    sql_lines.append(f"-- Time-Series training data generated: {datetime.now().strftime('%Y-%m-%d')}")
    sql_lines.append("-- Contains > 1000 rows with daily variability for analysis")
    sql_lines.append("-- --------------------------------------------------------")
    sql_lines.append("USE ventas;")
    sql_lines.append("")
    sql_lines.append("INSERT INTO ventas (fecha, hora, concepto, cantidad, precio, total) VALUES")

    values_list = []
    total_rows = 0

    for dia in range(NUM_DIAS):
        fecha_actual = FECHA_INICIO + timedelta(days=dia)

        # Simulate seasonality: fewer sales on weekends, random peaks
        es_finde = fecha_actual.weekday() >= 5

        # Number of sales per day (biased random)
        # Normal days: 3 to 8 sales. Weekends: 1 to 5 sales.
        if es_finde:
            num_ventas = random.randint(1, 5)
        else:
            num_ventas = random.randint(3, 9)

        for _ in range(num_ventas):
            # 1. Random time (09:00 to 19:00)
            hora_h = random.randint(9, 18)
            hora_m = random.randint(0, 59)
            hora_s = random.randint(0, 59)
            hora_str = f"{hora_h:02d}:{hora_m:02d}:{hora_s:02d}"
            fecha_str = fecha_actual.strftime('%Y-%m-%d')

            # 2. Choose product based on weight (probability)
            prod = random.choices(productos, weights=[p[2] for p in productos], k=1)[0]
            nombre = prod[0]
            precio = prod[1]

            # 3. Quantity (1 to 3, rarely more)
            cantidad = random.choices([1, 2, 3, 4], weights=[80, 15, 4, 1], k=1)[0]
            total = round(precio * cantidad, 2)

            # Format SQL row
            row = f"('{fecha_str}', '{hora_str}', '{nombre}', {cantidad}, {precio}, {total})"
            values_list.append(row)
            total_rows += 1

    # Join everything and close with a semicolon
    sql_lines.append(",\n".join(values_list) + ";")
    sql_lines.append(f"\n-- Total rows generated: {total_rows}")

    return "\n".join(sql_lines)


if __name__ == "__main__":
    # Generate and print the SQL script
    print(generar_sql())

