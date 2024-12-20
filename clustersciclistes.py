"""
@ IOC - CE IABD
@ Jordi Burgos
"""
import os
import logging
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

def load_dataset(path):
	"""
	Carrega el dataset de registres dels ciclistes

	arguments:
		path -- dataset

	Returns: dataframe
	"""
	return pd.read_csv(path, delimiter=',')

def EDA(df):
	"""
	Exploratory Data Analysis del dataframe

	arguments:
		df -- dataframe

	Returns: None
	"""
	logging.info('\n%s', df.shape)
	logging.info('\n%s', df[:5])
	logging.info('\n%s', df.columns)
	logging.info('\n%s', df.info())

def clean(df):
	"""
	Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

	arguments:
		df -- dataframe

	Returns: dataframe
	"""
	df = df.drop(columns=['dorsal'])
	logging.info('\n%s', df.columns)
	return df

def extract_true_labels(df):
	"""
	Guardem les etiquetes dels ciclistes (BEBB, ...)

	arguments:
		df -- dataframe

	Returns: numpy ndarray (true labels)
	"""
	true_labels = df['tipus_ciclista'].values
	df = df.drop(columns=['tipus_ciclista'])
	return true_labels

def visualitzar_pairplot(df):
	"""
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	arguments:
		df -- dataframe

	Returns: None
	"""
	grafic = sns.pairplot(df)
	try:
		os.makedirs(os.path.dirname('img/'))
		grafic.savefig('img/pairplot.png')
	except FileExistsError:
		pass

def clustering_kmeans(data, n_clusters=4):
	"""
	Crea el model KMCrea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el modeleans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	arguments:
		data -- les dades: tp i tb

	Returns: model (objecte KMeans)
	"""
	model = KMeans(n_clusters=n_clusters, random_state=42)
	model.fit(data)
	return model

def visualitzar_clusters(data, labels):
	"""
	Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

	arguments:
		data -- el dataset sobre el qual hem entrenat
		labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

	Returns: None
	"""
	try:
		os.makedirs(os.path.dirname('img/'))
	except FileExistsError:
		pass

	fig = plt.figure()
	logging.info('\nPrimeres files del dataset:\n%s', data.head())
	sns.scatterplot(x='temps_pujada', y='temps_baixada', data=data, hue=labels, palette='rainbow')
	plt.savefig('img/clusters.png')
	fig.clf()


def associar_clusters_patrons(tipus, model):
	"""
	Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
	S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

	arguments:
	tipus -- un array de tipus de patrons que volem actualitzar associant els labels
	model -- model KMeans entrenat

	Returns: array de diccionaris amb l'assignació dels tipus als labels
	"""
	# proposta de solució

	dicc = {'tp':0, 'tb': 1}

	logging.info('Centres:')
	for j in range(len(tipus)):
		logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(j, model.cluster_centers_[j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

	# Procés d'assignació
	ind_label_0 = -1
	ind_label_1 = -1
	ind_label_2 = -1
	ind_label_3 = -1

	suma_max = 0
	suma_min = 50000

	for j, center in enumerate(clustering_model.cluster_centers_):
		suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
		if suma_max < suma:
			suma_max = suma
			ind_label_3 = j
		if suma_min > suma:
			suma_min = suma
			ind_label_0 = j

	tipus[0].update({'label': ind_label_0})
	tipus[3].update({'label': ind_label_3})

	lst = [0, 1, 2, 3]
	lst.remove(ind_label_0)
	lst.remove(ind_label_3)

	if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
		ind_label_1 = lst[0]
		ind_label_2 = lst[1]
	else:
		ind_label_1 = lst[1]
		ind_label_2 = lst[0]

	tipus[1].update({'label': ind_label_1})
	tipus[2].update({'label': ind_label_2})

	logging.info('\nHem fet l\'associació')
	logging.info('\nTipus i labels:\n%s', tipus)
	return tipus

def generar_informes(df, tipus):
	"""
	Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
	4 fitxers de ciclistes per cadascun dels clústers

	arguments:
		df -- dataframe
		tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

	Returns: None
	"""
	df['label'] = clustering_model.labels_

	ciclistes_labels = [
		df[df['label'] == 0],
		df[df['label'] == 1],
		df[df['label'] == 2],
		df[df['label'] == 3]
	]

	for index in range(4):
		logging.debug('Ciclistes del clúster %s:\n%s\n', index, ciclistes_labels[index])
	
	try:
		os.makedirs(os.path.dirname('informes/'))
	except FileExistsError:
		pass
	for tipus_cluster in tipus:
		arxiu = tipus_cluster['name'].replace(' ', '_') + '.txt'
		t = [t for t in tipus if t['name'] == tipus_cluster['name']]
		ciclistes = ciclistes_labels[t[0]['label']].index
		with open(f'informes/{arxiu}', 'w', encoding='utf-8') as foutput:
			for ciclista in ciclistes:
				foutput.write(f'{ciclista}\n')

		foutput.close()

	logging.info('S\'han generat els informes en la carpeta informes/\n')

def nova_prediccio(dades, model):
	"""
	Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

	arguments:
		dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
		model -- clustering model
	Returns: (dades agrupades, prediccions del model)
	"""
	df_dades_ciclistes = pd.DataFrame(dades, columns=['dorsal', 'temps_pujada', 'temps_baixada'])
	df_dades_ciclistes = df_dades_ciclistes.drop(columns=['dorsal'])
	return df_dades_ciclistes, model.predict(df_dades_ciclistes)
# ----------------------------------------------

if __name__ == "__main__":

	logging.basicConfig(format='%(message)s', level=logging.INFO) # canviar entre DEBUG i INFO
	logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR) # per tal de què el matplotlib no vomiti molts missatges

	"""
	TODO:
	load_dataset
	EDA
	clean
	extract_true_labels
	eliminem el tipus, ja no interessa .drop('tipus', axis=1)
	visualitzar_pairplot
	clustering_kmeans
	pickle.dump(...) guardar el model
	mostrar scores i guardar scores
	visualitzar_clusters
	"""
	path_dataset = './data/ciclistes.csv'
	cursa_data = load_dataset(path_dataset)
	
	EDA(cursa_data)
	
	cursa_data = clean(cursa_data)
	
	true_labels = extract_true_labels(cursa_data)
	cursa_data = cursa_data.drop(columns=['tipus_ciclista'])

	visualitzar_pairplot(cursa_data)
	clustering_model = clustering_kmeans(cursa_data)
	
	pickle.dump(clustering_model, open('model/clustering_model.pkl', 'wb'))
	logging.info('Model KMeans guardat a model/clustering_model.pkl')
	logging.info('Scores:')
	logging.info('Homogeneïtat: %.3f', homogeneity_score(true_labels, clustering_model.labels_))
	logging.info('Completitud: %.3f', completeness_score(true_labels, clustering_model.labels_))
	logging.info('V-measure: %.3f', v_measure_score(true_labels, clustering_model.labels_))
	# Guardar scores en un archivo
	scores = {
		'homogeneity_score': homogeneity_score(true_labels, clustering_model.labels_),
		'completeness_score': completeness_score(true_labels, clustering_model.labels_),
		'v_measure_score': v_measure_score(true_labels, clustering_model.labels_)
	}

	with open('model/scores.pkl', 'wb') as f:
		pickle.dump(scores, f)

	logging.info('Scores guardats a model/scores.pkl')
	
	visualitzar_clusters(cursa_data, clustering_model.labels_)

	# array de diccionaris que assignarà els tipus als labels
	tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]
	tipus = associar_clusters_patrons(tipus, clustering_model)
	
	# Afegim la columna label al dataframe
	cursa_data['label'] = clustering_model.labels_
	logging.debug('\nColumna label:\n%s', cursa_data[:5])

	tipus = associar_clusters_patrons(tipus, clustering_model)
	# guardem la variable tipus
	with open('model/tipus_dict.pkl', 'wb') as f:
		pickle.dump(tipus, f)
	logging.info('\nTipus i labels:\n%s', tipus)
	
	# generem els informes
	generar_informes(cursa_data, tipus)
	
	# Classificació de nous valors
	nous_ciclistes = [
		[500, 3230, 1430], # BEBB
		[501, 3300, 2120], # BEMB
		[502, 4010, 1510], # MEBB
		[503, 4350, 2200] # MEMB
	]

	"""
	nova_prediccio

	#Assignació dels nous valors als tipus
	for i, p in enumerate(pred):
		t = [t for t in tipus if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
	"""
	logging.debug('\nNous valors:\n%s', nous_ciclistes)
	df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, clustering_model)
	logging.info('\nPredicció dels valors:\n%s', pred)
	
	for i, p in enumerate(pred):
		t = [t for t in tipus if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
