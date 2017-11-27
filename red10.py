import os, sys
import tensorflow as tf

#Paso 1
#Declarando el grado de mensajes que mandará la consola, el nivel 2 añade "WARNING" como filtro de los log error.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Paso 2
# Obtiene el valor ingresado por el usuario despues de la ejecución
# El argumento de ruta sería imagen.jpg despues de la instrucción normal de 
#python rede10.py dentro de la ruta indicada
ruta = sys.argv[1]

# Se lee la imagen dicha anteriormente con la ejecución del código
image_data = tf.gfile.FastGFile(ruta, 'rb').read()

#Paso 3
#Despues de leer la imagen y entrenarlas con retrain.py se crea un archivo "retrained_labels.txt"
#Este archivo lo que contiene son los nombres de los animales
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Se abre ya el grafo entrenado
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

#Paso 4
#Se abre una sesion de tensorflow, la neurona principal
with tf.Session() as sess:
    # Comprueba la imagen dada con los grafico y hace una predicción
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Ordena las predicciones en orden de puntuacion
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    #Almacena el nombre de la carpeta en human_string y la puntuacion obtenida en score, luego estas son almacenadas en arreglos
    #para su comparacion
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (Predicción = %.5f)' % (human_string, score))
    
    if human_string == "perros"
        os.system("start c:\dog.wav")