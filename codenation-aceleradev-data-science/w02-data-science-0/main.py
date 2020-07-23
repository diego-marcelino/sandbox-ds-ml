#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


csv_url = 'black_friday.csv'
black_friday = pd.read_csv(csv_url)


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    return black_friday.shape


# In[5]:


result = q1()
assert type(result) == tuple
assert len(result) == 2
result


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    shape = black_friday.loc[
        (black_friday['Gender'] == "F") & (black_friday['Age'] == '26-35')
    ].shape
    return int(shape[0])


# In[9]:


result = q2()
assert type(result) == int
result


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[10]:


def q3(feature='User_ID'):
    uniques = black_friday.nunique()
    return int(uniques[feature])


# In[11]:


result = q3()
assert type(result) == int
result


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[12]:


def q4():
    return black_friday.dtypes.nunique()


# In[13]:


result = q4()
assert type(result) == int
result


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[14]:


def q5():
    null_row_filter = pd.isnull(black_friday).any(axis=1)
    nulls_regs = black_friday[null_row_filter].shape[0]
    all_regs = black_friday.shape[0]
    return nulls_regs / all_regs


# In[15]:


result = q5()
assert type(result) == float
assert 0.0 <= result <= 1.0
result


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[16]:


def q6():
    null_count = black_friday.isnull().sum()
    null_count.sort_values(inplace=True, ascending=False)
    return int(null_count[0])


# In[17]:


result = q6()
assert type(result) == int
result


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[18]:


def q7(feature='Product_Category_3'):
    mode = black_friday[feature].mode(dropna=True)
    return mode[0]


# In[19]:


result = q7()
assert result
result


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[20]:


def normalize(feature):
    max_value = black_friday[feature].max()
    min_value = black_friday[feature].min()
    range_value = max_value - min_value
    feature_norm = (black_friday[feature] - min_value) / range_value
    return feature_norm


# In[21]:


def q8(feature = 'Purchase'):
    feature_norm = normalize(feature)
    return float(np.mean(feature_norm))


# In[22]:


result = q8()
assert type(result) == float
assert 0.0 <= result <= 1.0
result


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[23]:


def q9(feature = 'Purchase'):
    feature_norm = normalize(feature)
    filter = feature_norm.between(-1, 1, inclusive=True)
    return int(feature_norm[filter].count())


# In[24]:


result = q9()
assert type(result) == int
result


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[25]:


def q10():
    cat_2_null = black_friday['Product_Category_2'].isnull()
    cat_3_null = black_friday['Product_Category_3'][cat_2_null]
    return int(cat_3_null.count()) == 0


# In[26]:


result = q10()
assert type(result) == bool
result

