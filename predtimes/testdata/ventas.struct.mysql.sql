-- --------------------------------------------------------
-- Estructura de tabla para: ventas2
-- Basada en el esquema original de 'ventas'
-- --------------------------------------------------------

-- 1. Asegurarse de usar la base de datos correcta
USE ventas;

-- 2. Eliminar la tabla si ya existe (opcional, para reiniciar)
DROP TABLE IF EXISTS ventas2;

-- 3. Crear la tabla 'ventas2'
CREATE TABLE ventas2 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fecha DATE,
    hora TIME,
    concepto VARCHAR(255),
    cantidad INT,
    precio DECIMAL(10, 2),
    total DECIMAL(10, 2)
);
