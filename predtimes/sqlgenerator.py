import random
from datetime import datetime, timedelta

# Configuración
NUM_DIAS = 365*8  # Un año de datos
FECHA_INICIO = datetime(2000, 5, 1)
ARCHIVO_SALIDA = "ventas_entrenamiento.sql"

# Productos (Nombre, Precio, Peso de probabilidad de venta)
# El peso alto significa que se vende más frecuentemente (ej. Cables vs Laptops)
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
    sql_lines = []
    
    # Header del SQL
    sql_lines.append("-- --------------------------------------------------------")
    sql_lines.append(f"-- Datos de entrenamiento Time-Series generados: {datetime.now().strftime('%Y-%m-%d')}")
    sql_lines.append("-- Contiene > 1000 filas con variabilidad diaria para análisis")
    sql_lines.append("-- --------------------------------------------------------")
    sql_lines.append("USE ventas;")
    sql_lines.append("")
    sql_lines.append("INSERT INTO ventas (fecha, hora, concepto, cantidad, precio, total) VALUES")

    values_list = []
    
    total_rows = 0
    
    for dia in range(NUM_DIAS):
        fecha_actual = FECHA_INICIO + timedelta(days=dia)
        
        # Simular estacionalidad: Fin de semana vende menos, o días aleatorios con picos
        es_finde = fecha_actual.weekday() >= 5
        
        # Número de ventas por día (Random con sesgo)
        # Días normales: 3 a 8 ventas. Fines de semana: 0 a 4 ventas.
        if es_finde:
            num_ventas = random.randint(1, 5)
        else:
            num_ventas = random.randint(3, 9)
            
        for _ in range(num_ventas):
            # 1. Hora aleatoria (09:00 a 19:00)
            hora_h = random.randint(9, 18)
            hora_m = random.randint(0, 59)
            hora_s = random.randint(0, 59)
            hora_str = f"{hora_h:02d}:{hora_m:02d}:{hora_s:02d}"
            fecha_str = fecha_actual.strftime('%Y-%m-%d')

            # 2. Elegir producto basado en peso (probabilidad)
            prod = random.choices(productos, weights=[p[2] for p in productos], k=1)[0]
            nombre = prod[0]
            precio = prod[1]

            # 3. Cantidad (1 a 3, raramente más)
            cantidad = random.choices([1, 2, 3, 4], weights=[80, 15, 4, 1], k=1)[0]
            
            total = round(precio * cantidad, 2)

            # Formatear fila SQL
            row = f"('{fecha_str}', '{hora_str}', '{nombre}', {cantidad}, {precio}, {total})"
            values_list.append(row)
            total_rows += 1

    # Unir todo y cerrar con punto y coma
    sql_lines.append(",\n".join(values_list) + ";")
    
    sql_lines.append(f"\n-- Total filas generadas: {total_rows}")

    return "\n".join(sql_lines)

# Ejecutar y mostrar
print(generar_sql())
