#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns

from os import path, rename
from sklearn import preprocessing
from statsmodels.distributions.empirical_distribution import ECDF
from urllib.request import  urlretrieve
from zipfile import ZipFile


# In[2]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[5]:


def q1():
    Q1, Q2, Q3 = 0.25, 0.5, 0.75
    norm_quantis = dataframe['normal'].quantile([Q1, Q2, Q3])
    binom_quantis = dataframe['binomial'].quantile([Q1, Q2, Q3])
    zip_quantis = zip(norm_quantis, binom_quantis)
    return tuple(round(q[0] - q[1], 3) for q in zip_quantis)


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[6]:


def q2():
    serie = dataframe['normal']
    x_ = serie.mean()
    s = serie.std()
    interval_min = x_ - s
    interval_max = x_ + s
    ecdf = ECDF(serie)
    interval = ecdf(interval_max) - ecdf(interval_min)
    return float(round(interval, 3))


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[7]:


def q3():
    m_norm, v_norm = dataframe['normal'].mean(), dataframe['normal'].var()
    m_binom, v_binom = dataframe['binomial'].mean(), dataframe['binomial'].var()
    return tuple(np.round([m_binom - m_norm, v_binom - v_norm], 3))


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[8]:


# just to get the data file
if not path.isfile('pulsar_stars.csv'):
    DATA_ZIP_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip'
    dump_header = ','.join([f'h{n}'for n in range(9)])
    urlretrieve (DATA_ZIP_URL, 'data.zip')
    with ZipFile('data.zip', 'r') as zf:
        zf.extractall()

    rename('HTRU_2.csv', 'pulsar_stars.csv')
    with open('pulsar_stars.csv', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(dump_header + '\n' + content)


# In[9]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[11]:


false_pulsar_df = stars[stars.target == False]
false_pulsar_mean_prof = false_pulsar_df.mean_profile.values.reshape(-1, 1)
scaler = preprocessing.StandardScaler()
false_pulsar_mean_profile_standardized = scaler.fit_transform(
    false_pulsar_mean_prof
).flatten()


# In[12]:


def q4():
    quantis_points = [0.8, 0.9, 0.95]
    quantis = sct.norm.ppf(quantis_points, loc=0, scale=1)
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    probabilities = ecdf(quantis)
    return tuple(round(p, 3) for p in probabilities)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[13]:


def q5():
    quantis_points = [0.25, 0.5, 0.75]
    normal_quantis = sct.norm.ppf(quantis_points, loc=0, scale=1)
    mean_profile_quantis = np.quantile(
        false_pulsar_mean_profile_standardized, quantis_points)
    zip_quantis = zip(mean_profile_quantis, normal_quantis)
    return tuple(round(q[0] - q[1], 3) for q in zip_quantis)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
