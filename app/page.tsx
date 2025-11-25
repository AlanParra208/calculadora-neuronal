"use client";
import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Calculadora() {
  // --- 1. Estados con Tipado de TypeScript ---
  const [modeloActual, setModeloActual] = useState<tf.LayersModel | null>(null);
  const [operacion, setOperacion] = useState<'suma' | 'resta'>('suma'); 
  const [valA, setValA] = useState<string>('');
  const [valB, setValB] = useState<string>('');
  const [resultado, setResultado] = useState<string | null>(null);
  const [cargando, setCargando] = useState<boolean>(false);
  const [origenModelo, setOrigenModelo] = useState<string>(''); // Para mostrar si es Archivo o Memoria

  // --- 2. Función Auxiliar: Crear Modelo en Memoria (Fallback) ---
  // Esto simula un modelo ya entrenado para que la app funcione sin archivos externos
  const crearModeloEnMemoria = (op: 'suma' | 'resta'): tf.LayersModel => {
    const model = tf.sequential();
    // Una sola neurona es suficiente para suma/resta lineal
    model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
    
    // Establecemos los pesos manualmente para simular que el modelo "aprendió"
    // Suma: Salida = 1*A + 1*B + 0
    // Resta: Salida = 1*A - 1*B + 0
    const w1 = 1;
    const w2 = op === 'suma' ? 1 : -1;
    
    model.setWeights([
      tf.tensor2d([[w1], [w2]]), // Pesos (Kernel)
      tf.tensor1d([0])           // Sesgo (Bias)
    ]);
    
    return model;
  };

  // --- 3. Efectos (Carga del Modelo) ---
  useEffect(() => {
    async function cambiarModelo() {
      setModeloActual(null); 
      setResultado(null);
      setCargando(true);
      setOrigenModelo('');
      
      const path = operacion === 'suma' 
        ? '/modelo_suma/model.json' 
        : '/modelo_resta/model.json';
      
      try {
        // Intentamos cargar el archivo (esto funcionará en producción si los archivos existen)
        // Usamos window.location.origin para evitar errores de URL relativa en blobs
        const baseURL = typeof window !== 'undefined' ? window.location.origin : '';
        const fullPath = `${baseURL}${path}`;
        
        console.log(`Intentando cargar desde: ${fullPath}`);
        
        // Nota: En este entorno demo, esto probablemente fallará (404 o error de parseo)
        // porque los archivos no existen. El catch lo manejará.
        const m = await tf.loadLayersModel(path);
        
        setModeloActual(m);
        setOrigenModelo('Archivo JSON');
        console.log(`Modelo de ${operacion} cargado desde archivo.`);

      } catch (err) {
        console.warn("No se pudo cargar el archivo externo (esperado en demo). Generando modelo en memoria...", err);
        
        // FALLBACK: Generar modelo sintético en el navegador
        const mSintetico = crearModeloEnMemoria(operacion);
        setModeloActual(mSintetico);
        setOrigenModelo('Generado en Memoria (Demo)');
      } finally {
        setCargando(false);
      }
    }

    cambiarModelo();
  }, [operacion]);

  // --- 4. Lógica de Predicción ---
  const calcular = async () => {
    if (!modeloActual || valA === '' || valB === '') return;

    const nA = parseFloat(valA);
    const nB = parseFloat(valB);
    
    if (isNaN(nA) || isNaN(nB)) {
        alert("Por favor ingresa números válidos");
        return;
    }

    let inputTensor: tf.Tensor2D | null = null;
    let prediccionTensor: tf.Tensor | tf.Tensor[] | null = null;

    try {
      inputTensor = tf.tensor2d([[nA, nB]]);
      
      prediccionTensor = modeloActual.predict(inputTensor);
      
      if (Array.isArray(prediccionTensor)) {
        const data = await prediccionTensor[0].data();
        setResultado(data[0].toFixed(2));
      } else {
        const data = await (prediccionTensor as tf.Tensor).data();
        setResultado(data[0].toFixed(2));
      }

    } catch (error) {
      console.error("Error en el cálculo:", error);
      setResultado("Error");
    } finally {
      if (inputTensor) inputTensor.dispose();
      
      if (prediccionTensor) {
        if (Array.isArray(prediccionTensor)) {
            prediccionTensor.forEach(t => t.dispose());
        } else {
            (prediccionTensor as tf.Tensor).dispose();
        }
      }
    }
  };

  // --- 5. Renderizado (JSX) ---
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 p-4 font-sans">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        
        <h1 className="text-2xl font-bold text-center text-gray-800 mb-2">
          Calculadora Neuronal TS
        </h1>
        
        {/* Indicador de Origen del Modelo */}
        <div className="flex justify-center mb-6">
            <span className={`text-xs px-2 py-1 rounded-full ${
                cargando ? 'bg-yellow-100 text-yellow-800' :
                origenModelo.includes('Memoria') ? 'bg-purple-100 text-purple-800' : 'bg-green-100 text-green-800'
            }`}>
                Estado: {cargando ? 'Cargando...' : origenModelo || 'Listo'}
            </span>
        </div>

        <div className="flex justify-center gap-4 mb-6">
          <button
            onClick={() => setOperacion('suma')}
            className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
              operacion === 'suma' 
                ? 'bg-blue-600 text-white shadow-md' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Suma (+)
          </button>
          
          <button
            onClick={() => setOperacion('resta')}
            className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
              operacion === 'resta' 
                ? 'bg-red-500 text-white shadow-md' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Resta (-)
          </button>
        </div>

        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Valor A</label>
            <input
              type="number"
              placeholder="0"
              value={valA}
              onChange={(e) => setValA(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition text-black"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Valor B</label>
            <input
              type="number"
              placeholder="0"
              value={valB}
              onChange={(e) => setValB(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition text-black"
            />
          </div>
        </div>

        <button
          onClick={calcular}
          disabled={!modeloActual || cargando}
          className={`w-full py-3 px-6 rounded-lg font-bold text-lg text-white transition-all ${
            modeloActual 
              ? 'bg-gray-900 hover:bg-gray-800 shadow-md hover:shadow-lg' 
              : 'bg-gray-400 cursor-not-allowed'
          }`}
        >
          {cargando ? 'Cargando modelo...' : 'Calcular con IA'}
        </button>

        {resultado !== null && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg text-center animate-fade-in">
            <p className="text-sm text-green-600 font-semibold uppercase tracking-wide">
              Predicción del Modelo
            </p>
            <p className="text-4xl font-bold text-green-800 mt-1">
              {resultado}
            </p>
            {origenModelo.includes('Memoria') && (
                <p className="text-xs text-gray-400 mt-2 italic">
                    (Calculado usando modelo sintético en navegador)
                </p>
            )}
          </div>
        )}
        
      </div>
    </div>
  );
}