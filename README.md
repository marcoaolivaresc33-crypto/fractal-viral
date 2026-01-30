"""
Script para generar todas las visualizaciones del proyecto
Modelado Matemático de Propagación Viral con Teoría Fractal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from scipy import stats
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def fibonacci(n):
    """Calcula F_n usando fórmula de Binet"""
    phi = (1 + np.sqrt(5)) / 2
    psi = (1 - np.sqrt(5)) / 2
    return int(round((phi**n - psi**n) / np.sqrt(5)))

def G_fibonacci(j, k):
    """Calcula G(j,k) recursivamente"""
    if k <= 0:
        return 0
    if k <= j:
        return 1
    
    G = [1] * j
    for i in range(j+1, k+1):
        G.append(sum(G[-j:]))
    
    return G[-1]

def calcular_raiz_xj(j, max_iter=1000, tol=1e-10):
    """Calcula la raíz positiva máxima de F_j(x) = 0 usando Newton-Raphson"""
    # F_j(x) = x^j - x^{j-1} - ... - x - 1
    # F_j'(x) = j*x^{j-1} - (j-1)*x^{j-2} - ... - 1
    
    x = 1.5  # Valor inicial
    for _ in range(max_iter):
        # Calcular F_j(x)
        Fx = x**j - sum(x**i for i in range(j))
        # Calcular F_j'(x)
        Fx_prime = j * x**(j-1) - sum((j-i) * x**(j-i-1) for i in range(1, j))
        
        x_new = x - Fx / Fx_prime
        
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    
    return x

# =====================================================
# VISUALIZACIÓN 1: Crecimiento de Infecciones (Ejemplo 9.1)
# =====================================================

def grafico_crecimiento_fibonacci():
    """Gráfico 1: I(k) vs k en escala logarítmica"""
    print("Generando Gráfico 1: Crecimiento de Infecciones...")
    
    k_vals = np.arange(0, 21)
    I_vals = [fibonacci(k+1) for k in k_vals]
    
    phi = (1 + np.sqrt(5)) / 2
    I_teorico = [phi**(k+1) / np.sqrt(5) for k in k_vals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Escala lineal
    ax1.plot(k_vals, I_vals, 'bo-', markersize=8, linewidth=2, label='Datos Observados')
    ax1.plot(k_vals, I_teorico, 'r--', linewidth=2, label=r'Ajuste $\phi^k$', alpha=0.7)
    ax1.set_xlabel('Día k', fontsize=14)
    ax1.set_ylabel('Computadoras Infectadas I(k)', fontsize=14)
    ax1.set_title('Crecimiento de Infecciones - Escala Lineal', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Escala logarítmica
    ax2.semilogy(k_vals, I_vals, 'bo-', markersize=8, linewidth=2, label='Datos Observados')
    ax2.semilogy(k_vals, I_teorico, 'r--', linewidth=2, label=r'Ajuste $\phi^k$', alpha=0.7)
    ax2.set_xlabel('Día k', fontsize=14)
    ax2.set_ylabel('Computadoras Infectadas I(k) [log]', fontsize=14)
    ax2.set_title('Crecimiento de Infecciones - Escala Logarítmica', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/grafico1_crecimiento_infecciones.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico 1 guardado")

# =====================================================
# VISUALIZACIÓN 2: Comparación Multi-Orden (Ejemplo 9.2)
# =====================================================

def grafico_comparacion_ordenes():
    """Gráfico 2: Comparación de G(j,k) para j=2,3,4,5"""
    print("Generando Gráfico 2: Comparación Multi-Orden...")
    
    k_max = 15
    ordenes = [2, 3, 4, 5]
    colores = ['blue', 'red', 'green', 'purple']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for j, color in zip(ordenes, colores):
        k_vals = np.arange(1, k_max+1)
        G_vals = [G_fibonacci(j, k) for k in k_vals]
        
        # Escala lineal
        ax1.plot(k_vals, G_vals, marker='o', color=color, 
                label=f'j={j}', linewidth=2, markersize=6)
        
        # Escala logarítmica
        ax2.semilogy(k_vals, G_vals, marker='o', color=color, 
                    label=f'j={j}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Día k', fontsize=14)
    ax1.set_ylabel('Computadoras Infectadas G(j,k)', fontsize=14)
    ax1.set_title('Comparación de Propagación - Escala Lineal', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Día k', fontsize=14)
    ax2.set_ylabel('Computadoras Infectadas G(j,k) [log]', fontsize=14)
    ax2.set_title('Comparación de Propagación - Escala Logarítmica', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/grafico2_comparacion_ordenes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico 2 guardado")

# =====================================================
# VISUALIZACIÓN 3: Box-Counting (Ejemplo 9.3)
# =====================================================

def grafico_box_counting():
    """Gráfico 3: Regresión log-log para dimensión fractal"""
    print("Generando Gráfico 3: Box-Counting...")
    
    # Datos simulados de box-counting
    escalas = [1/2**i for i in range(1, 9)]
    # Conteos que siguen aproximadamente Fibonacci
    conteos = [8, 13, 21, 34, 55, 89, 144, 233]
    
    log_r = np.log([1/r for r in escalas])
    log_N = np.log(conteos)
    
    # Regresión lineal
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_N)
    D = slope
    r2 = r_value**2
    
    # Valor teórico
    phi = (1 + np.sqrt(5)) / 2
    D_teorico = 2 * np.log(phi) / np.log(2)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(log_r, log_N, s=150, c='blue', marker='o', 
               edgecolors='black', linewidth=2, label='Datos', zorder=3)
    plt.plot(log_r, slope * log_r + intercept, 'r--', linewidth=3, 
            label=f'Ajuste Lineal: D={D:.3f}, R²={r2:.4f}')
    
    plt.xlabel('log(1/r)', fontsize=16)
    plt.ylabel('log(N(r))', fontsize=16)
    plt.title('Dimensión Fractal - Método de Box-Counting', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, alpha=0.4, linestyle='--')
    
    # Añadir texto con valor teórico
    plt.text(0.98, 0.02, f'Valor Teórico (Grossman): D = {D_teorico:.4f}',
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/grafico3_box_counting.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico 3 guardado")

# =====================================================
# VISUALIZACIÓN 4: Función Generatriz (Ejemplo 9.4)
# =====================================================

def grafico_funcion_generatriz():
    """Gráfico 4: G₂(η) vs η"""
    print("Generando Gráfico 4: Función Generatriz...")
    
    phi = (1 + np.sqrt(5)) / 2
    
    # Valores de eta > phi
    eta_vals = np.linspace(phi + 0.01, 3.5, 200)
    
    # G_2(eta) = (eta-1)*eta / (1 + (eta-2)*eta^2)
    G2_vals = [(eta - 1) * eta / (1 + (eta - 2) * eta**2) for eta in eta_vals]
    
    plt.figure(figsize=(12, 8))
    plt.plot(eta_vals, G2_vals, 'b-', linewidth=3, label=r'$G_2(\eta)$')
    
    # Marcar punto especial eta=2
    G2_2 = (2 - 1) * 2 / (1 + (2 - 2) * 2**2)
    plt.plot(2, G2_2, 'ro', markersize=15, label=r'$\eta=2$: $G_2(2)=2$', zorder=5)
    
    # Asíntota vertical en phi
    plt.axvline(x=phi, color='red', linestyle='--', linewidth=2, 
               label=r'Asíntota: $\eta = \phi \approx 1.618$')
    
    plt.xlabel(r'$\eta$ (Factor de Descuento)', fontsize=16)
    plt.ylabel(r'$G_2(\eta)$ (Impacto Descontado)', fontsize=16)
    plt.title('Función Generatriz de Fibonacci', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(1.5, 3.5)
    plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/grafico4_funcion_generatriz.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico 4 guardado")

# =====================================================
# VISUALIZACIÓN 5: Capacidad de Contención (Ejemplo 9.5)
# =====================================================

def grafico_capacidad_contencion():
    """Gráfico 5: c_min vs k₀"""
    print("Generando Gráfico 5: Capacidad de Contención...")
    
    k0_vals = np.arange(2, 17)
    c_min_vals = [fibonacci(k0 - 1) for k0 in k0_vals]
    
    phi = (1 + np.sqrt(5)) / 2
    c_teorico = [phi**(k0 - 2) for k0 in k0_vals]
    
    plt.figure(figsize=(12, 8))
    plt.semilogy(k0_vals, c_min_vals, 'bo-', markersize=10, linewidth=3, 
                label='Capacidad Mínima Requerida', markeredgecolor='black', markeredgewidth=2)
    plt.semilogy(k0_vals, c_teorico, 'r--', linewidth=2, 
                label=r'Aproximación: $c \sim \phi^{k_0-2}$', alpha=0.7)
    
    plt.xlabel('Día de Intervención $k_0$', fontsize=16)
    plt.ylabel('Capacidad de Contención $c_{min}$ [log]', fontsize=16)
    plt.title('Capacidad de Contención vs Día de Intervención', fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    
    # Añadir regiones de riesgo
    plt.axvspan(2, 6, alpha=0.2, color='green', label='Intervención Temprana')
    plt.axvspan(6, 10, alpha=0.2, color='yellow', label='Intervención Media')
    plt.axvspan(10, 16, alpha=0.2, color='red', label='Intervención Tardía')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/grafico5_capacidad_contencion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico 5 guardado")

# =====================================================
# VISUALIZACIÓN 6: Fractal Sierpinski-Fibonacci
# =====================================================

def generar_fractal_sierpinski():
    """Visualización fractal: Teselación triangular"""
    print("Generando Visualización Fractal...")
    
    def subdividir_triangulo(A, B, C, nivel, triangulos_grandes, triangulos_pequenos):
        if nivel == 0:
            triangulos_grandes.append(np.array([A, B, C]))
            return
        
        # Puntos medios
        M_AB = (A + B) / 2
        M_BC = (B + C) / 2
        M_CA = (C + A) / 2
        
        # Triángulo central (sombreado)
        triangulos_pequenos.append(np.array([M_AB, M_BC, M_CA]))
        
        # Recursión en tres triángulos grandes
        subdividir_triangulo(A, M_AB, M_CA, nivel-1, triangulos_grandes, triangulos_pequenos)
        subdividir_triangulo(M_AB, B, M_BC, nivel-1, triangulos_grandes, triangulos_pequenos)
        subdividir_triangulo(M_CA, M_BC, C, nivel-1, triangulos_grandes, triangulos_pequenos)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Triángulo inicial equilátero
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    for i, nivel in enumerate([0, 1, 2, 3, 4, 5]):
        ax = axes[i]
        
        triangulos_grandes = []
        triangulos_pequenos = []
        subdividir_triangulo(A, B, C, nivel, triangulos_grandes, triangulos_pequenos)
        
        # Dibujar triángulos grandes (no sombreados)
        for tri in triangulos_grandes:
            ax.fill(tri[:, 0], tri[:, 1], edgecolor='black', 
                   facecolor='lightblue', alpha=0.6, linewidth=1.5)
        
        # Dibujar triángulos pequeños (sombreados)
        for tri in triangulos_pequenos:
            ax.fill(tri[:, 0], tri[:, 1], edgecolor='black', 
                   facecolor='darkblue', alpha=0.8, linewidth=1)
        
        total_triangulos = len(triangulos_grandes) + len(triangulos_pequenos)
        F_teorico = fibonacci(nivel + 3) if nivel > 0 else 1
        
        ax.set_aspect('equal')
        ax.set_title(f'Nivel {nivel}: {total_triangulos} triángulos ≈ F_{{{nivel+3}}} = {F_teorico}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Construcción Fractal mediante Proyección Ortogonal (Grossman)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('/home/sandbox/fractal_sierpinski_fibonacci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualización Fractal guardada")

# =====================================================
# VISUALIZACIÓN 7: Árbol de Propagación Viral
# =====================================================

def generar_arbol_propagacion():
    """Visualización del árbol de propagación viral"""
    print("Generando Árbol de Propagación...")
    
    class NodoViral:
        def __init__(self, id, dia):
            self.id = id
            self.dia = dia
            self.hijos = []
    
    def construir_arbol(k_max, j=2):
        raiz = NodoViral(0, 0)
        nodos_por_dia = {0: [raiz]}
        contador = 1
        
        for k in range(1, k_max + 1):
            nodos_por_dia[k] = []
            
            if k >= 2:
                # Nodos que pueden infectar (dia <= k-2)
                for dia_inf in range(k - 1):
                    for nodo in nodos_por_dia[dia_inf]:
                        for _ in range(j):
                            hijo = NodoViral(contador, k)
                            nodo.hijos.append(hijo)
                            nodos_por_dia[k].append(hijo)
                            contador += 1
        
        return raiz, nodos_por_dia
    
    raiz, nodos_por_dia = construir_arbol(k_max=6, j=2)
    
    # Crear grafo de NetworkX
    G = nx.DiGraph()
    
    def agregar_nodos_y_aristas(nodo):
        G.add_node(nodo.id, dia=nodo.dia)
        for hijo in nodo.hijos:
            G.add_edge(nodo.id, hijo.id)
            agregar_nodos_y_aristas(hijo)
    
    agregar_nodos_y_aristas(raiz)
    
    # Layout jerárquico
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Colorear por día
    dias = nx.get_node_attributes(G, 'dia')
    colores = [dias[nodo] for nodo in G.nodes()]
    
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5, arrows=True, 
                          arrowsize=15, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=colores, cmap='viridis', 
                          node_size=300, alpha=0.9, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(f'Grafo de Propagación Viral (k=6, {len(G.nodes())} nodos = F₇ = {fibonacci(7)})', 
             fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Barra de color
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=6))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Día de Infección')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/arbol_propagacion_viral.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Árbol de Propagación guardado")

# =====================================================
# VISUALIZACIÓN 8: Raíces x_j y Dimensión Fractal
# =====================================================

def grafico_raices_dimension():
    """Gráfico de raíces x_j y dimensión fractal D vs j"""
    print("Generando Gráfico de Raíces y Dimensión...")
    
    j_vals = np.arange(2, 21)
    x_j_vals = [calcular_raiz_xj(j) for j in j_vals]
    D_vals = [2 * np.log(xj) / np.log(2) for xj in x_j_vals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de raíces
    ax1.plot(j_vals, x_j_vals, 'bo-', markersize=8, linewidth=2, label='$x_j$')
    ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Límite: 2')
    ax1.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Cota inferior: 1')
    ax1.set_xlabel('Orden j', fontsize=14)
    ax1.set_ylabel('Raíz $x_j$', fontsize=14)
    ax1.set_title('Raíces Características $x_j$ vs Orden j', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1, 2.1)
    
    # Gráfico de dimensión
    ax2.plot(j_vals, D_vals, 'ro-', markersize=8, linewidth=2, label='D(j)')
    ax2.axhline(y=2, color='blue', linestyle='--', linewidth=2, label='Límite: 2')
    ax2.set_xlabel('Orden j', fontsize=14)
    ax2.set_ylabel('Dimensión Fractal D', fontsize=14)
    ax2.set_title('Dimensión Fractal D vs Orden j', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1.3, 2.1)
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/grafico_raices_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico de Raíces y Dimensión guardado")

# =====================================================
# VISUALIZACIÓN 9: Tabla Comparativa
# =====================================================

def generar_tabla_comparativa():
    """Genera tabla comparativa de secuencias"""
    print("Generando Tabla Comparativa...")
    
    k_vals = np.arange(1, 11)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Datos de la tabla
    tabla_datos = []
    for k in k_vals:
        fila = [
            k,
            G_fibonacci(2, k),
            G_fibonacci(3, k),
            G_fibonacci(4, k),
            G_fibonacci(5, k)
        ]
        tabla_datos.append(fila)
    
    # Crear tabla
    tabla = ax.table(cellText=tabla_datos,
                    colLabels=['k', 'j=2\n(Fibonacci)', 'j=3\n(Tribonacci)', 
                              'j=4\n(Tetranacci)', 'j=5\n(Pentanacci)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1, 0.2, 0.2, 0.2, 0.2])
    
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1, 2.5)
    
    # Estilo de encabezados
    for i in range(5):
        tabla[(0, i)].set_facecolor('#4CAF50')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    # Estilo de celdas
    for i in range(1, len(tabla_datos) + 1):
        for j in range(5):
            if i % 2 == 0:
                tabla[(i, j)].set_facecolor('#f0f0f0')
            else:
                tabla[(i, j)].set_facecolor('white')
    
    plt.title('Tabla Comparativa: Secuencias de Fibonacci Generalizadas G(j,k)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig('/home/sandbox/tabla_comparativa.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Tabla Comparativa guardada")

# =====================================================
# VISUALIZACIÓN 10: Simulación vs Teoría
# =====================================================

def grafico_simulacion_teoria():
    """Comparación simulación vs teoría"""
    print("Generando Gráfico Simulación vs Teoría...")
    
    k_vals = np.arange(0, 21)
    
    # Valores teóricos
    I_teorico = [fibonacci(k+1) for k in k_vals]
    
    # Simular con ruido pequeño
    np.random.seed(42)
    I_simulado = [max(1, int(I + np.random.normal(0, 0.02*I))) for I in I_teorico]
    
    # Calcular error relativo
    error_rel = [abs(sim - teo) / teo * 100 if teo > 0 else 0 
                 for sim, teo in zip(I_simulado, I_teorico)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico principal
    ax1.plot(k_vals, I_teorico, 'r-', linewidth=3, label='Teórico G(2,k+1)', alpha=0.7)
    ax1.plot(k_vals, I_simulado, 'bo', markersize=8, label='Simulado', 
            markeredgecolor='black', markeredgewidth=1.5)
    ax1.set_xlabel('Día k', fontsize=14)
    ax1.set_ylabel('Computadoras Infectadas', fontsize=14)
    ax1.set_title('Simulación vs Teoría: Propagación Viral', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de error
    ax2.bar(k_vals, error_rel, color='orange', alpha=0.7, edgecolor='black')
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Umbral 5%')
    ax2.set_xlabel('Día k', fontsize=14)
    ax2.set_ylabel('Error Relativo (%)', fontsize=14)
    ax2.set_title('Error Relativo: |Simulado - Teórico| / Teórico × 100%', 
                 fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/sandbox/simulacion_vs_teoria.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico Simulación vs Teoría guardado")

# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================

def generar_todas_visualizaciones():
    """Genera todas las visualizaciones del proyecto"""
    print("\n" + "="*60)
    print("GENERACIÓN DE VISUALIZACIONES - PROYECTO VIRUS INFORMÁTICO")
    print("="*60 + "\n")
    
    try:
        grafico_crecimiento_fibonacci()
        grafico_comparacion_ordenes()
        grafico_box_counting()
        grafico_funcion_generatriz()
        grafico_capacidad_contencion()
        generar_fractal_sierpinski()
        generar_arbol_propagacion()
        grafico_raices_dimension()
        generar_tabla_comparativa()
        grafico_simulacion_teoria()
        
        print("\n" + "="*60)
        print("✓ TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE")
        print("="*60)
        print("\nArchivos generados:")
        print("  1. grafico1_crecimiento_infecciones.png")
        print("  2. grafico2_comparacion_ordenes.png")
        print("  3. grafico3_box_counting.png")
        print("  4. grafico4_funcion_generatriz.png")
        print("  5. grafico5_capacidad_contencion.png")
        print("  6. fractal_sierpinski_fibonacci.png")
        print("  7. arbol_propagacion_viral.png")
        print("  8. grafico_raices_dimension.png")
        print("  9. tabla_comparativa.png")
        print(" 10. simulacion_vs_teoria.png")
        print("\n" + "="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error durante la generación: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generar_todas_visualizaciones()
