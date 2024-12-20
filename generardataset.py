import logging
import numpy as np

def generar_dataset(num, ind, dict):
	"""
	Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del dictionari
	TODO: completar arguments, return. num és el número de files/ciclistes a 
	generar. ind és l'index/identificador/dorsal.
	"""
	data = []
	# Per a cada ciclista generem un temps de pujada i un temps de baixada
	for i in range(num):
		logging.basicConfig(level=logging.INFO)
		logging.info("Generant dades per al ciclista %s", i + 1)
		ciclista = dict[i % len(dict)]
		temps_pujada = int(np.random.normal(ciclista["mu_p"], ciclista["sigma"]))
		temps_baixada = int(np.random.normal(ciclista["mu_b"], ciclista["sigma"]))
		logging.info("Ciclista %s - Temps pujada: %s s, Temps baixada: %s s", ciclista['name'], temps_pujada, temps_baixada)
		data.append([ind + i, ciclista["name"], temps_pujada, temps_baixada])
	return data

if __name__ == "__main__":

	STR_CICLISTES = 'data/ciclistes.csv'


	# BEBB: bons escaladors, bons baixadors
	# BEMB: bons escaladors, mal baixadors
	# MEBB: mal escaladors, bons baixadors
	# MEMB: mal escaladors, mal baixadors

	# Port del Cantó (18 Km de pujada, 18 Km de baixada)
	# pujar a 20 Km/h són 54 min = 3240 seg
	# pujar a 14 Km/h són 77 min = 4268 seg
	# baixar a 45 Km/h són 24 min = 1440 seg
	# baixar a 30 Km/h són 36 min = 2160 seg
	MU_P_BE = 3240 # mitjana temps pujada bons escaladors
	MU_P_BE = 4268 # mitjana temps pujada mals escaladors
	MU_B_BB = 1440 # mitjana temps baixada bons baixadors
	MU_B_MB = 2160 # mitjana temps baixada mals baixadors
	SIGMA = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
		{"name":"BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
		{"name":"MEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
		{"name":"MEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA}
	]

	NUM_CICLISTES = 200
	IND_INICIAL = 1  

	data = generar_dataset(NUM_CICLISTES, IND_INICIAL, dicc)


	with open(STR_CICLISTES, "w", encoding="utf-8") as foutput:
		foutput.write("dorsal,tipus_ciclista,temps_pujada,temps_baixada\n")

	for d in data:
		foutput.write(f"{d[0]},{d[1]},{d[2]},{d[3]}\n")
	
	foutput.close()

	logging.info("s'ha generat data/ciclistes.csv")
