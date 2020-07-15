


import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
import numpy as np
import scipy as sp
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns



caddy = pd.read_csv('caddy_jz.csv')
caddy = caddy.rename(columns={'ObjectPrice': 'target'})		# kaller y for target
caddy.replace([np.inf, -np.inf], np.nan, inplace = True)	# finner alle -inf og inf verdier og setter de til nan
caddy.dropna(axis='rows', inplace = True)					# fjerner alle rader med nan

# print(caddy.head())
# print(caddy.columns)


# sns.heatmap(caddy.corr(), annot=True)						# numerertkorrelasjon plottet
# plt.show()
# sns.pairplot(caddy)											# vanlig korrelasjonsplot
# plt.show()


# Deler i trenings og test sett
target = caddy["target"]
caddy.drop(columns=['target'], inplace = True)
X_train, X_test, y_train, y_test = train_test_split(caddy, target, test_size=.3)





# Her er det jeg tenker er en rask mAte og fA seg en oversikt over om det er noe som bQr gjQres med forklaringsvariablene.
# Om det sA er en funksjon, som roten, til variablen, eller om det er interaksjonsvariabel som bQr vAEre med. Det ser litt rotete ut
# og kunne sikkert blitt gjort penere ved A legge inn funksjoner man vil ha med og navnene pA de in hver sin liste som man sA looper over.
# Uansett, jeg skjekker ogsA for collinAEritet, men har ikke tenkt pA variance inflation factor(VIF) som gAr ut pA at 3 forskjellige
# faktorer kan ha en collinAEritet seg i mellom. I dette tilfelle hvor du virker som du tenker du vil ha en grei modell og 
# vise til bruktbilselgere, tenker jeg det her likevel er viktig, selv om det kanskje ikke er alle variablene som popper opp man 
# trenger A ta med. Det dukker likevel opp enkelte faktorer som gir fint mening som typ effekt/engineVolume, og det kan man jo fint
# forklare til en bruktbilselger. Det samme tenker jeg om det er noe som har en logaritmisk utvikling eller hva. Det er i hvertfall
# enklere A ha gode variabler, og sA tenke seg til hvorfor de antageligvis har en sammenheng med prisen :)



def lag_nye_var(X_train, X_test, y_train, n_break_points = 10, max_collinearity = 0.7, min_linearity_with_y = 0.35, max_forklaringsvariabler = 40, give_X_names = None, plot = None):

	columns = list(X_train.columns)
	# Her lages funksjoner av forklaringsvariablene, og da har jeg bare tatt med de vanligste man ofte ser
	for j in range(len(X_train.T)):

		#FOR TRAIN
		X_train[X_train.iloc[:,j].name+"sq"] = X_train.iloc[:,j].transform(lambda x: x**(2))
		X_train[X_train.iloc[:,j].name+"qb"] = X_train.iloc[:,j].transform(lambda x: x**(3))
		X_train[X_train.iloc[:,j].name+"root"] = X_train.iloc[:,j].transform(lambda x: x**(1/2))
		X_train[X_train.iloc[:,j].name+"log"] = X_train.iloc[:,j].transform(lambda x: np.log(x))
		X_train[X_train.iloc[:,j].name+"exp"] = X_train.iloc[:,j].transform(lambda x: np.exp(x))

		#FOR TEST
		X_test[X_test.iloc[:,j].name+"sq"] = X_test.iloc[:,j].transform(lambda x: x**(2))
		X_test[X_test.iloc[:,j].name+"qb"] = X_test.iloc[:,j].transform(lambda x: x**(3))
		X_test[X_test.iloc[:,j].name+"root"] = X_test.iloc[:,j].transform(lambda x: x**(1/2))
		X_test[X_test.iloc[:,j].name+"log"] = X_test.iloc[:,j].transform(lambda x: np.log(x))
		X_test[X_test.iloc[:,j].name+"exp"] = X_test.iloc[:,j].transform(lambda x: np.exp(x))

	# Her lages interaksjonsvariablene, og da er det, for at det ikke skal bli alt for mange og krunglete A forstA hva de er,
	# laget av en kombinasjon av de vi hadde helt i starte med de vi nettopp har laget. AltsA ikke rooten av en ting ganget roten
	# roten av noe annet, men kun roten av noe ganget med for eksempel effekt. Uansett, her kan man egentlig gjQre som man vil
	# sA lenge man orker A forklare det etterpA tenker jeg!
	columns2 = list(X_train.columns)	
	for j in range(len(columns)):
		for i in range(len(columns2)):
			if columns[j] != X_train.iloc[:,i].name:

				#FOR TRAIN
				X_train[columns[j]+"+"+columns2[i]] = X_train[columns[j]] + X_train[columns2[i]]
				X_train[columns[j]+"-"+columns2[i]] = X_train[columns[j]] - X_train[columns2[i]]
				X_train[columns[j]+"*"+columns2[i]] = X_train[columns[j]] * X_train[columns2[i]]
				X_train[columns[j]+"/"+columns2[i]] = X_train[columns[j]] / X_train[columns2[i]]

				#FOR TEST
				X_test[columns[j]+"+"+columns2[i]] = X_test[columns[j]] + X_test[columns2[i]]
				X_test[columns[j]+"-"+columns2[i]] = X_test[columns[j]] - X_test[columns2[i]]
				X_test[columns[j]+"*"+columns2[i]] = X_test[columns[j]] * X_test[columns2[i]]
				X_test[columns[j]+"/"+columns2[i]] = X_test[columns[j]] / X_test[columns2[i]]

		print(j/len(columns))



	# FQr vi gjQr linAEr sjekk og collinAEr sjekk, legger vi til target for linAEr sjekken, og fjerner alle kolloner som har rare verdier.
	X_train['target'] = y_train
	X_train.replace([np.inf, -np.inf], np.nan, inplace = True)
	X_train.dropna(axis='columns', inplace = True)
	X_test.replace([np.inf, -np.inf], np.nan, inplace = True)
	X_test.dropna(axis='columns', inplace = True)


	# sA det her er ogsA en litt morsom mAte og tenke pA forklaringsvariablene pA, selv om vi har en linAEr model. Det er at
	# vi deler opp prisen (target) i n_intervaller, og sjekker om det er noen som forklaringsvariabler som er bedre pA de forskjellige
	# intervallene. Det er litt inspirert av inntekts modeller, hvor man egentlig bQr lage to modeller, en for de som tjener normalt
	# og en for de som tjener enormt mye. Det burde man i hvertfall i linAEre modeller, fordi man ikke klarer A forklare de 
	# som tjener enormt mye ellers. Dette er i sA mAte en forenkling, fordi det blir ikke like gode resultater som om vi lager 
	# egene modeller for outliersene, men det kan gi en god forbedring likevel :) 
	antall_intervaller = n_break_points			# Antall intervaller
	lengde = np.sqrt( (np.max(y_train) - np.min(y_train))**2 )/ antall_intervaller		# Lengde, pris1 til pris2, per intervall

	all_indexes = []	# Array for A legge inn indexene til forklaringsvariablene vi har lyst til A bruke

	for k in range(antall_intervaller):

		# Finner indexene som stemmer overens med pris intervallet vi er i
		interval_split = np.where(np.logical_and(y_train>=np.min(y_train)+(lengde*k), y_train<np.min(y_train)+(lengde*(k+1))))[0]
		mid_X_train = np.array(X_train)[interval_split]
		mid_X_train = pd.DataFrame(mid_X_train)

		all_diff_correlations = np.absolute(np.array(mid_X_train.corr())) # Finner korrelasjons matrisen


		train_linear_corr = all_diff_correlations[:-1,-1]		# Siden vi la til target til slutt litt lenger opp her til slutt, 
		# sA vet vi at det er siste linje vi vil ha i matrisen
		train_linear_corr = np.nan_to_num(train_linear_corr) 	# Av en eller annen grunn kommer rare verdier som max verdier,
		# sA her gjQr vi de om til 0
		index_of_best_linear_relation = train_linear_corr.argsort()[::-1]	# Og sA sorterer vi, og finner indexene

		if train_linear_corr[index_of_best_linear_relation][0] > 0:		# Skulle det vAEre slik at ingen har korrelasjon stopper vi med en gang
			index_of_good_fit = [index_of_best_linear_relation[0]]	# Ellers starter vi med A legge inn indexen til forklaringsvariablene som har mest linAEritet med prisen
			count = 1
			# Enkel loop som stopper om vi har nAdd maxforklaringsvariabel for intervallet, vi har gAtt gjennom alle, eller vi har kommet til forklaringsvariabler
			# som ligger under i verdi av hva vi ser pA som akseptabelt (min_linearity_with_y)
			while len(index_of_good_fit) < int(max_forklaringsvariabler/antall_intervaller) and count != len(index_of_best_linear_relation)\
			 and train_linear_corr[index_of_best_linear_relation[count]] > min_linearity_with_y:
				wrong = 0
				# sA sjekker vi kjapt om forklaringsvariablen k, har mindre collinAEritet med en av de andre som er valgt for intervallet (max_collinearity)
				for k in range(len(index_of_good_fit)):
					if wrong == 0:
						value = np.absolute(all_diff_correlations[index_of_good_fit[k],index_of_best_linear_relation[count]])
						if value > max_collinearity:
							wrong = 1
							break
				if wrong == 0:
					index_of_good_fit.append(index_of_best_linear_relation[count])
				count += 1
			all_indexes.append(index_of_good_fit)


	print("Antall forklaringsvariabler vi har endt opp med : ", len(np.unique(np.concatenate(all_indexes))))
	column_names_of_best_fit = np.array(X_train.columns)[np.unique(np.concatenate(all_indexes))]

	X_train = X_train[column_names_of_best_fit]
	X_test = X_test[column_names_of_best_fit]

	# Printer forklaringsvariablene om man Qnsker
	if give_X_names != None:
		print(column_names_of_best_fit)
	# Plotter om man vil
	if plot != None:
		X_train['target'] = y_train
		sns.heatmap(X_train.corr(), annot=True)
		plt.show()
		sns.pairplot(X_train)
		plt.show()
		X_train.drop(columns=['target'], inplace = True)

	return(X_train, X_test, y_train)

X_train, X_test, y_train = lag_nye_var(X_train, X_test, y_train, n_break_points = 6, \
	max_collinearity = 0.75, min_linearity_with_y = 0.5, max_forklaringsvariabler = 30, give_X_names = 1, plot = None)



# SA kommer det en del som jeg trengte A gjQre. Det er ikke nQdvendig A skalere forklaringsvariablene, men det lQnner seg! 
# Grunnen til det tror jeg er fordi nAr vi bruker pymc3 til linAEr regresjon sA lQser vi det ikke analytisk, vi tilnAErmer oss, og da skjer det raskere med 
# gode verdier for PC'en. Ikke gjQr det hvis det ikke blir nQdvendig, men gAr det sApass mye raskere, sA kan man jo fint regne ut verdiene vi fAr for betane 
# ved A skalere de motsatt av hva som skjer i sklearn preprocessing! Det som jeg ikke kom utenom var A skalere prisene pA bilene, og det tror jeg kommer av
# at jeg ikke vet hvordan jeg skal stille inn verdiene for priori modell riktig for hQyere verdier. Rettere sagt jeg har ikke prQvd, for det er kanskje 
# bare A skalere priori mean'ene med det jeg har skalert prisene ned? RETTELSE, Jeg har vel stilt inn priori verdiene slik at de passer til skalerte X variabler,
# sA de er skalerte likevel. Men ved A vie ut/bytte intervall på priori verdiene gAr det ann A komme seg unna det og skalere verdiene.


from sklearn import preprocessing

x1 = X_train.values #returns a numpy array
x2 = X_test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled1 = min_max_scaler.fit_transform(x1)
x_scaled2 = min_max_scaler.fit_transform(x2)
X_train = pd.DataFrame(x_scaled1)
X_test = pd.DataFrame(x_scaled2)

y_train = y_train / 10000
y_test = y_test / 10000




# Jeg lar de tingene jeg har prQvd ligge igjen, sA du kan se litt hva jeg har prøvd ut, og kanskje du finner en bedre lQsning selv :D
# Skal forklare hvorfor jeg har valgt det jeg har gjort ogsA

linear_model = pm.Model()

with linear_model: 
	# Priors for unknown model parameters

	# Her er standardavvikene vi kommer til A trenge underveis. Ofte bruker man HalfNormal, og det er fordi da 
	# trenger vi ikke sA mye optimalisering siden den har tetthet nAErmere en verdi, 0, uansett. Jeg gikk for en sA Apen modell
	# som mulig for A gjQre et eksempel ut av at man fint kan gjQre det i denne settingen. Det gAr raskt likevel!
	# OgsA er standardavvikene aldri under null :)

	# sigma = pm.HalfNormal('sigma', sd=10) # you could also try with a HalfCauchy that has longer/fatter tails
	sigma = pm.Uniform('sigma', lower=0, upper=5) # you could also try with a HalfCauchy that has longer/fatter tails
	# sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
	# sigma2 = pm.Uniform('sigma2', lower=0, upper=2)
	sigma2 = pm.Uniform('sigma2', lower=0, upper=30)
	# sigma2 = pm.HalfNormal('sigma2', sd=50)
	sigma3 = pm.Uniform('sigma3', lower=0, upper=10) # you could also try with a HalfCauchy that has longer/fatter tails
	# sigma3 = pm.HalfCauchy('sigma3', sd=10) # you could also try with a HalfCauchy that has longer/fatter tails


	# Som du vil se sA konvergerer ikke nu verdiene spesielt bra, og det veit jeg ikke hvorfor. Det kan tenkes at, 
	# siden dette er frihetsgradene studentT distrubusjonen, sA utgjQr det ikke spesielt stor rolle. Kanskje noen av disse rett og slett
	# ikke burde vAEre studentT priori. Og det stemmer ganske godt overens med alpha som om man plotter det ser mer ut som en versjon av 
	# gamma distrubusjonen. Jeg lar det vAEre siden det funker :P
	nu = pm.Uniform('nu', lower=20, upper=90)
	# nu = pm.HalfNormal('nu', sd=1)
	nu2 = pm.Uniform('nu2', lower=20, upper=90)
	# nu2 = pm.HalfNormal('nu2', sd=5)
	nu3 = pm.Uniform('nu3', lower=20, upper=90)


	# Alpha her er interception, og Beta er Betaverdiene.
	# Her tenker jeg du kan gjQre litt om du ikke Qnsker A skalere de forklaringsvariablene du ender opp med A bruke. Fordi som du ser
	# sendes det her bare in en shape, siden jeg ikke vet hvor mange betas vi har endt opp med pA forhAnd, og alle fAr mean = 0, og samme standardavvik.
	# Finner du forklaringsvariablene du vil ha pA forhAnd kan du lage egene betas for hver enkelt, noe som sikkert vil gi bedre resultater med litt
	# tweaking. Da kan du ogsA bruke denne som mal hvis du liker hvordan dette ser ut bedre https://docs.pymc.io/notebooks/GLM.html , under Hierarchical GLM
	alpha = pm.StudentT("alpha", mu=y_train.mean(),sd=sigma, nu = nu)
	# alpha = pm.Normal("alpha", mu=y_train.mean(),sd=sigma)
	# alpha = pm.Gamma('alpha', sd=sigma, mu=y_train.mean())
	betas = pm.StudentT("betas", mu=0,sd=sigma2, nu = nu2, shape=np.shape(X_train)[1])
	# betas = pm.Normal("betas", mu=0,sd=sigma2, shape=np.shape(X_train)[1])
	# alpha = pm.Gamma('lambda', alpha=sigma, beta=y_train.mean())


	# betas = pm.Normal("betas", mu=0,sd=sigma2, shape=np.shape(X_train)[1])
	# betas = pm.Uniform('betas', -3, 3, shape=np.shape(X_train)[1])	# Eventuelt gjQre dette og bruke hQyere n under trace og fit for at den skal selv
	# finne gode verdier

	# Her er altsA den linAEre modelle satt sammen 
	mu = alpha + pm.math.dot(betas, X_train.T)

	# Og sA sier jeg at modellen igjen er StudenT distrubuert, eller noe annet om du vil.
	# likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y_train)
	likelihood = pm.StudentT("likelihood", mu=mu, sd=sigma3, nu = nu3, observed=y_train)
	# likelihood = pm.Gamma("likelihood", mu=mu, sd=sigma3, observed=y_train)
	# likelihood = pm.Normal("likelihood", mu=mu, sd=sigma3, observed=y_train)
	# likelihood = pm.NormalMixture('likelihood', nu3, mu, tau=sigma3, observed=y_train)


	# pA min PC funker det ikke men som du ser er det noen som har brukt flere cores. Da mA man vel bruke GPU'en og slikt.
	# trace = pm.fit(1000, step, cores = 2)
	trace = pm.fit(method=pm.ADVI(), n=100000)


# Dette er et plot for A se om modellen konvergerer mot noe. Den skal ikke ende opp som en rett strekk likevel.
advi_elbo = pd.DataFrame(
    {'log-ELBO': -np.log(trace.hist),
     'n': np.arange(trace.hist.shape[0])})

_ = sns.lineplot(y='log-ELBO', x='n', data=advi_elbo)
plt.show()



# Her trekker vi posterior fra modellen vAr i trace.sample
advi_trace = trace.sample(10000)

# Dette plottet er for A se hvordan det har gAtt. hmm, det kan hvis du er usikker sA skal jeg forklare de bedre for det er enklere. Noen av plottene
# her tyder pA at vi kan gjQre det en del bedre. Som et par av nu verdiene og sd verdiene
pm.traceplot(advi_trace)
plt.show()
print(pm.summary(advi_trace))




# Dette er et fint plot som viser om posterioren vi trekker fra ligner pA distrubusjonen til bilprisene. Vet ikke om man kanskje burde brukt
# y_test?, men fra det jeg har sett har de brukt y_train her
ppc = pm.sample_posterior_predictive(advi_trace, samples=5000, model=linear_model)

# sns.kdeplot(y_train, alpha=0.5, lw=4, color='b')
sns.kdeplot(y_test, alpha=0.5, lw=4, color='b')
for i in range(100):
    sns.kdeplot(ppc['likelihood'][i], alpha=0.1, color='g')

plt.show()


# Her tar vi rett og slett gjennomsnittet av intercept (alpha) og betaene vAre for A sjekke om vi har lagd en ok modell. Dette kan selvfQlgelig brukes
# til A lage distrubusjonsplot av hver enkelt verdi, samt konfidensintervaller til hver verdi, eller sette det sammen og fA konfidens for predikasjonen vAr
chain = advi_trace[2000:]
alpha_pred = chain['alpha'].mean()
betas_pred = chain['betas'].mean(axis=0)
y_pred = alpha_pred + np.dot(betas_pred, X_test.T)


# Plotter for A sjekke, y_pred vs y_test, som burde gi en god indikasjon pA hvordan vi har gjort det
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# just for fun!
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(np.min(y_test), np.max(y_test), np.mean(y_test))

result = ((y_pred - y_test) / y_test*100).round()

print('Mean Absolute Error: %.1f percent.' % abs(result).mean())


# hAper det kan vAEre til hjelp, og inspirasjon! Vil du lage noen egene variabler, sA har jeg tenkt etter A ha sett plottene dine,
# at man kanskje burde ha en model for biler som kjQrer under x antall km i Aret, og en for biler som kjQrer over x antall km i Aret.
# Litt som inntekts modellen jeg nevnte. Andre variabler som kan vAEre spennende er "Ar siden utgivelse av bilmerke" (Er tesla mer ustabil enn Mercedes),
# og "Ar siden utgivelse av bil modell" (Er Golf mer stabil enn mange av de andre). Det er ting jeg kanskje ville forsQkt A fAtt med. Aja! Det er jo 
# en del kategoriske variabler her, som jeg ikke helt husker om man kanskje bQr modellere pA en annen mAte enn det som er gjort. Uansett det har funket
# sA det er vel en vurdering du fAr ta :D





